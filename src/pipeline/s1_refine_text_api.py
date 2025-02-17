import json
import torch
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

    def __init__(self, api):
        # self.api_client = APIClient(api)
        self.api_client = OpenAI(api_key="sk-0111a295a2774391af08dffd746d335b", base_url="https://api.deepseek.com")
        # self.model_name = self.api_client.available_models[0]
        self.model_name = 'deepseek-chat'
    @staticmethod
    def call_api(api_client: APIClient, model, image=None, prompt: dict = None):

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

    def summary(self, label, caption, refs, title, abstract):

        user = self.USER_PROMPT.format(label=label, caption=caption, refs=refs, title=title, abstract=abstract)
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


def process_txt(augmentor, figure_txt, fig_root, done_images, output):
    if not figure_txt.get('flg'):
        return
        
    image_name = figure_txt.get('name')
    image_label = figure_txt.get('label')
    image_path = figure_txt.get('path')
    if image_name in done_images or not os.path.exists(image_path):
        return

    caption = figure_txt.get('caption')
    ref = figure_txt.get('ref', [])
    title = figure_txt.get('paper_title')
    abstract = figure_txt.get('paper_abstract')
    refs = "\n\n".join(ref)

    response = augmentor.summary(image_label, caption, refs, title, abstract)
    if not response:
        return

    match = re.search(r"<[Ss]ummary>(.*?)</[Ss]ummary>", response, re.DOTALL)
    if match:
        figure_txt['summary'] = match.group(1).strip()
        save_output(output, figure_txt)
    else:
        print("No find summary")


def save_output(output_file, data):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(api, data, output, fig_root):
    lines = load_data(data)
    done_images = load_done_images(output)
    augmentor = TxtAugmenter(api)
    print("---------------------------Moedel loading----------------------------")
    for figure_txt in tqdm(lines):

        process_txt(augmentor, figure_txt, fig_root, done_images, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=f'http://0.0.0.0:23334', type=str)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--data", default="fig_txt_20241001-20241231.jsonl", type=str)
    parser.add_argument("--output", default="fig_txt_s_20241001-20241231.jsonl", type=str)
    parser.add_argument("--fig-root", default="single_images/", type=str)
    args = parser.parse_args()

    model = args.model
    api = args.api
    data = os.path.join(DATA_OUTPUT_DIR, args.data)
    output = os.path.join(DATA_OUTPUT_DIR, args.output)
    fig_root = os.path.join(DATA_OUTPUT_DIR, args.fig_root)
    print("---------------------------Start----------------------------")
    main(api, data, output, fig_root)
