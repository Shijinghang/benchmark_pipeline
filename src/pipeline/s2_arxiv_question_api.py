import json
import re
import os
import argparse
from tqdm import tqdm
from lmdeploy.serve.openai.api_client import APIClient
import sys
from openai import OpenAI
import io
import base64

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

    USER_PROMPT = """image description: {summary}"""

    def __init__(self, api):
        # self.api_client = APIClient(api)
        # self.api_client = OpenAI(
        #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        #     api_key="sk-7d5b1bd59b9d4db7a82b907169ab3a35", # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        self.api_client = OpenAI(api_key="sk-0111a295a2774391af08dffd746d335b", base_url="https://api.deepseek.com")
        # self.model_name = self.api_client.available_models[0]
        self.model_name = 'deepseek-chat'

    @staticmethod
    def call_api(api_client, model, image=None, prompt: dict = None):

        messages, content = [], []

        if system := prompt.get('system'):
            messages.append({"role": "system", "content": system})

        if image:
            # 将PIL图像转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})
        if user := prompt.get("user"):
            content.append({"type": "text", "text": user})

        messages.append({"role": "user", "content": content})

        # responds = api_client.chat_completions_v1(model, messages, temperature=0, max_tokens=8296)
        responds = api_client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=8192, stream=False)
        #for respond in responds:
        print(responds)
        result = responds.choices[0].message.content
        return result

    def generate(self, summary):

        user = self.USER_PROMPT.format(summary=summary)
        try:
            output = self.call_api(self.api_client, model=self.model_name, prompt={'system': self.SYSTEM_PROMPT, 'user': user})
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

def process_txt(genertor, figure_txt, done_images, output):

    image_name = figure_txt.get('name')
    if image_name in done_images:
        return

    summary = figure_txt.get('summary')

    response = genertor.generate(summary)
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


def main(api, data, output, fig_root):
    lines = load_data(data)
    done_images = load_done_images(output)
    genertor = QuestionGenertor(api)
    print("---------------------------Moedel loading----------------------------")
    for figure_txt in tqdm(lines):
        if not figure_txt.get('flg'):
            continue
        process_txt(genertor, figure_txt, done_images, output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=f'http://0.0.0.0:23334', type=str)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", type=str)
    parser.add_argument("--data", default="fig_txt_s_20241001-20241231.jsonl", type=str)
    parser.add_argument("--output", default="arxiv_test.jsonl", type=str)
    parser.add_argument("--fig-root", default="single_images/", type=str)
    args = parser.parse_args()

    model = args.model
    api = args.api
    data = os.path.join(DATA_OUTPUT_DIR, args.data)
    output = os.path.join(DATA_OUTPUT_DIR, args.output)
    fig_root = os.path.join(DATA_OUTPUT_DIR, args.fig_root)
    print("---------------------------Start----------------------------")
    main(api, data, output, fig_root)
