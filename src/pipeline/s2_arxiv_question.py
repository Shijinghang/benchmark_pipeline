import json
import re
import os
import argparse
from tqdm import tqdm
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig,PytorchEngineConfig
import sys

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
Provide the question, correct answer, source reference, and relevant image description in structured XML format:

<response>
    <result>
        <question>Question text here.</question>
        <options>
	        <A>...</A>
			<B>...</B>
			<C>...</C>
			...
        </options>
        <answer>Correct option here.</answer>
        <source>Correct option reference in the description.</source>
    </result>
    or 
    <result>Unable to generate question</result>
</response>
"""

  USER_PROMPT = """image description: {summary}"""

  def __init__(self, model, tp=8):
    backend_config = PytorchEngineConfig(cache_max_entry_count=0.1,tp=tp)
    self.gen_config = GenerationConfig(temperature=0, max_new_tokens=8192)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def generate(self, summary):
    content = self.USER_PROMPT.format(summary=summary)
    messages = [
        {
            "role": "system",
            "content": self.SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": content
        },
    ]
    try:
      output = self.pipeline(messages, gen_config=self.gen_config).text
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
    return {
        "question": question_text,
        "options": options_dict,
        "answer": correct_answer,
        "source": source_text
    }


def process_image(genertor, figure_txt, done_images, output):
  image_name = figure_txt.get('name')
  if image_name in done_images:
      return

  summary = figure_txt.get('summary')

  response = genertor.generate(summary)
  if not response:
      return
  
  response = genertor.generate(summary)
  if not response:
    return


#   match = re.search(r"{(.)*['\"]summary['\"]:(.)*}", response, re.DOTALL)
  match = extract_question_data( response)

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


def main(model, data, output, fig_root, tp):
  lines = load_data(data)
  done_images = load_done_images(output)
  generator = QuestionGenertor(model=model, tp=tp)
  print("---------------------------Moedel loading----------------------------")
  for figure_txt in tqdm(lines):
    if not figure_txt.get('flg'):
        continue
    process_image(generator, figure_txt, fig_root, done_images, output)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3-FP8", type=str)
  parser.add_argument("--data", default="fig_txt_s_20241001-20241231.jsonl", type=str)
  parser.add_argument("--output", default="arxiv_test.jsonl", type=str)
  parser.add_argument("--fig-root", default="single_images/", type=str)
  parser.add_argument("--tp", default=8, type=int)
  args = parser.parse_args()

  model = args.model
  data = os.path.join(DATA_OUTPUT_DIR, args.data)
  output = os.path.join(DATA_OUTPUT_DIR, args.output)
  fig_root = os.path.join(DATA_OUTPUT_DIR, args.fig_root)
  tp = args.tp

  print("---------------------------Start----------------------------")
  main(model, data, output, fig_root, tp)

