import json
import re
import os
import argparse
from tqdm import tqdm
import sys

# 根据模式选择性导入
try:
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    from lmdeploy.serve.openai.api_client import APIClient
except ImportError:
    pass

from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *


class TxtAugmenter:
    SYSTEM_PROMPT = """You are a highly skilled research assistant specializing in summarizing image information from academic papers. You will receive the image's label, caption, and the paragraphs in the paper that mention the image. Additionally, you will be provided with the paper's title and abstract as background information.

Your task is to summarize the information about the image indicated by the label based on the information provided. The summary includes but is not limited to the background, the image itself, the conclusion, etc. (if mentioned in the given content). Your summary will serve senior scholars; please describe it in a formal and scholarly manner, ensuring clarity, quality, and comprehensiveness.

**Formatting and Content Restrictions:**
    - Strictly base your summary on the information provided. Do not add personal interpretations or explanations beyond what is explicitly stated in the given text.
    - Ensure all LaTeX formats are deleted, with the exception of mathematical formulas.
    - When you refer to the target image, use expressions such as "the image" or "the figure," instead of "Figure~\\ref{?}" or "Figure ?"

**Feedback Mechanism**:
    - If the provided information about the image is unclear or insufficient to generate a summary, respond with Case1.
    - If the content is irrelevant to the target image, also respond with Case2.

**Output Format:**
Please summarize the image information and output the result in XML format with the tag <summary>."""

    USER_PROMPT = """label: {label}
caption: {caption}
paragraphs: {refs}
title: {title}
abstract: {abstract}"""

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            return response.choices[0].message.content

    def summary(self, label, caption, refs, title, abstract):
        user_prompt = self.USER_PROMPT.format(
            label=label, caption=caption, 
            refs=refs, title=title, abstract=abstract
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            if self.use_api:
                output = self._call_api(messages)
            else:
                output = self._call_local(messages)
                
            return output
        except Exception as e:
            print(f"Error generating text: {e}")
            return None


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]


def load_done_images(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return [json.loads(line)['name'] for line in f]
    return []


def process_item(augmentor, item, done_images, output_path):
    identifier = item.get('name', item.get('label', ''))
    if identifier in done_images:
        return

    required_fields = ['label', 'caption', 'ref', 'paper_title', 'paper_abstract']
    if not all(item.get(field) for field in required_fields):
        return

    response = augmentor.summary(
        item['label'],
        item['caption'],
        "\n\n".join(item['ref']),
        item['paper_title'],
        item['paper_abstract']
    )

    if not response:
        return

    match = re.search(r"<[Ss]ummary>(.*?)</[Ss]ummary>", response, re.DOTALL)
    if match:
        item['summary'] = match.group(1).strip()
        save_output(output_path, item)
    else:
        print(f"No summary found for {identifier}")


def save_output(output_path, data):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(config):
    lines = load_data(config['data_path'])
    done_images = load_done_images(config['output_path'])
    
    augmentor = TxtAugmenter(config)
    print("Model/API initialized" + (" (API)" if config['use_api'] else " (Local)"))
    
    for item in tqdm(lines):
        process_item(augmentor, item, done_images, config['output_path'])


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
        # 'top_k': args.top_k
        
        
    }

    # 参数验证
    if config['use_api']:
        if not config['api_base']:
            raise ValueError("API base URL is required in API mode")
    else:
        if not config['model']:
            raise ValueError("Model path is required in local mode")

    main(config)