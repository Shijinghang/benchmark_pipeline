import json
import re
import os
import argparse
from tqdm import tqdm
import sys
import io
import base64

# 根据模式选择性导入
try:
	from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
	from lmdeploy.serve.openai.api_client import APIClient
except ImportError:
	pass

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *


class QuestionGenertor:
	SYSTEM_PROMPT = """You are a skilled assistant specializing in generating high-quality academic questions based on image descriptions from academic papers. Your task is to create a multiple choice question that effectively assesses the examinee's ability to apply professional knowledge and analyze the image.

**Feedback Mechanism**:  
Check if the description includes both the visual details and the conclusion derived from the image. If either is missing, respond with "Unable to generate question" and do not proceed.

**Formatting and Content Restrictions**:  
- **Question**:  
  1. The question should require higher-level analysis, such as interpreting results or drawing inferences based on the image’s data and conclusions, rather than focusing on visual attributes.  
	 - **Qualified Example**: What are the main differences in redshift distribution between LRDs and AGNs shown in the image?  
	 - **Unqualified Example**: What does the purple bar represent shown in the image?  
  2. Avoid unnecessary background information and professional knowledge that may give away the answer. The question should require scientific reasoning, not direct recall of the description.

- **Answer**:  
  1. The correct answer must be supported directly by the information in the image description. The model should clearly indicate the reference to the description that supports the answer.
  2. Contains several plausible but incorrect options, No need to provide reference.

**Output Format**:  
Provide the question, options, correct answer, correct option reference in structured XML format:


<result>
	<question>Question text here.</question>
	<opitons>
		<A>Option A here.</A>
		<B>Option B here.</B>
		<C>Option C here.</C>
		<D>Option D here.</D>
	</options>
	<answer>Correct option here.</answer>
	<source>Correct option reference in the description.</source>
</result>
or 
<result>Unable to generate question</result>

"""

	USER_PROMPT = """image description: {text}"""

	def __init__(self, config):
		self.use_api = config['use_api']
		self.model = config['model']

		self.max_tokens = config['max_tokens']
		self.max_new_tokens = config['max_new_tokens']
		self.temperature = config['temperature']

		if self.use_api:
			# API模式初始化
			if config['api_type'] == 'openai':
				self.client = OpenAI(api_key=config['api_key'], base_url=config['api_base'])
			elif config['api_type'] == 'lmdeploy':
				self.client = APIClient(config['api_base'])
			else:
				raise ValueError(f"Unsupported API type: {config['api_type']}")
		else:
			# 本地模式初始化
			backend_config = TurbomindEngineConfig(tp=config['tp'])
			self.gen_config = GenerationConfig(temperature=self.temperature, max_new_tokens=self.max_new_tokens)
			self.pipeline = pipeline(
				self.model,
				backend_config=backend_config,
			)

	@staticmethod
	def make_messages(image=None, prompt: dict = None):

		messages, content = [], []

		if system := prompt.get('system'):
			messages.append({"role": "system", "content": system})

		if user := prompt.get("user"):
			content.append({"type": "text", "text": user})

		if image:
			# 将PIL图像转换为base64
			buffered = io.BytesIO()
			image.save(buffered, format="PNG")
			img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
			content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})

		messages.append({"role": "user", "content": content})
		return messages

	def _call_local(self, messages):
		return self.pipeline(messages, gen_config=self.gen_config).text

	def _call_api(self, messages):
		if hasattr(self.client, 'chat_completions_v1'):
			# 处理lmdeploy的API调用
			response = self.client.chat_completions_v1(
				self.model,
				messages,
				temperature=self.temperature,
				max_tokens=self.max_tokens,
			)
			return response[0].text
		else:
			# 处理OpenAI格式的API调用
			response = self.client.chat.completions.create(model=self.model,
														   messages=messages,
														   temperature=self.temperature,
														   max_tokens=self.max_tokens,
														   stream=False)
			return response.choices[0].message.content

	def generate(self, text):
		user = self.USER_PROMPT.format(text=text)
		messages = self.make_messages(prompt={'system': self.SYSTEM_PROMPT, 'user': user})

		try:
			if self.use_api:
				output = self._call_api(messages)
			else:
				output = self._call_local(messages)
			return output
		except Exception as e:
			print(f"Error generating text: {e}")
			return None


def load_data(data):
	with open(data, "r", encoding="utf-8") as f:
		return [json.loads(line) for line in f.readlines()]


def load_done_images(output):
	if os.path.exists(output):
		with open(output, "r", encoding="utf-8") as f:
			return [json.loads(l)['name'] for l in f.readlines()]
	return []


def extract_question_data(xml_response):
	# 定义正则表达式
	question_pattern = r'<question>(.*?)</question>'
	options_pattern = r'<([A-Z])>(.*?)</\1>'  # 捕获选项 A, B, C 等
	answer_pattern = r'<answer>(.*?)</answer>'
	source_pattern = r'<source>(.*?)</source>'
	unable_to_generate_pattern = r'<result>Unable to generate question</result>'

	# 如果匹配到 "Unable to generate question"，则直接返回 False
	if re.search(unable_to_generate_pattern, xml_response):
		return False

	# 提取问题
	question = re.search(question_pattern, xml_response)
	if not question:
		return False  # 如果没有找到问题，则返回 False
	question_text = question.group(1)

	# 提取选项
	options = re.findall(options_pattern, xml_response)
	if not options:
		return False  # 如果没有选项，则返回 False
	options_dict = {opt[0]: opt[1] for opt in options}

	# 提取正确答案
	answer = re.search(answer_pattern, xml_response)
	if not answer:
		return False  # 如果没有找到答案，则返回 False
	correct_answer = answer.group(1)

	# 提取来源
	source = re.search(source_pattern, xml_response)
	if not source:
		return False  # 如果没有找到来源，则返回 False
	source_text = source.group(1)

	# 返回提取的数据
	return {"question": question_text, "options": options_dict, "answer": correct_answer, "source": source_text}


def process_txt(genertor, figure_txt, output):
	text = figure_txt.get('text')
	response = genertor.generate(text)
	if not response:
		return

	match = extract_question_data(response)
	if match:
		figure_txt['question'] = match['question']
		for option in match['options'].keys():
			figure_txt[option] = match['options'][option]
		figure_txt['answer'] = match['answer']
		figure_txt['source'] = match['source']
		save_output(output, figure_txt)
	else:
		print("No find summary")


def save_output(output_file, data):
	with open(output_file, "a", encoding="utf-8") as f:
		f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(config):
	lines = load_data(config['data_path'])
	done_images = load_done_images(config['output_path'])
	genertor = QuestionGenertor(config)
	print("---------------------------Moedel loading----------------------------")
	for figure_txt in tqdm(lines):
		if figure_txt.get('name') in done_images:
			continue
		process_txt(genertor, figure_txt, config['output_path'])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# 公共参数
	parser.add_argument("--data", required=True, help="Input data file")
	parser.add_argument("--output", required=True, help="Output file")

	# 模式选择
	parser.add_argument("--use-api", action="store_true", help="Use API mode")

	# API相关参数
	parser.add_argument("--api-base", help="API base URL")
	parser.add_argument("--api-key", help="API key")
	parser.add_argument("--api-type", choices=['lmdeploy', 'openai'], default='lmdeploy')

	# 本地模型参数
	parser.add_argument("--model", default="deepseek-chat", help="Local model path")
	parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism")

	# 模型参数
	parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum length of the input sequence")
	parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
	# parser.add_argument("--top-p", type=float, default=0.7, help="Top-p sampling parameter")
	# parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling parameter")

	args = parser.parse_args()

	config = {
		'use_api': args.use_api,
		'data_path': os.path.join(DATA_OUTPUT_DIR, args.data),
		'output_path': os.path.join(DATA_OUTPUT_DIR, args.output),

		# API配置
		'api_base': args.api_base,
		'api_key': args.api_key,
		'api_type': args.api_type,

		# 本地模型配置
		'model': args.model,
		'tp': args.tp,

		# 模型参数
		'max_tokens': args.max_tokens,
		'max_new_tokens': args.max_new_tokens,
		'temperature': args.temperature,
	}

	# 参数验证
	if config['use_api']:
		if not config['api_base']:
			raise ValueError("API base URL is required in API mode")
	else:
		if not config['model']:
			raise ValueError("Model path is required in local mode")

	main(config)
