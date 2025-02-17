import json
import torch
import re
import os
import argparse
from tqdm import tqdm
from lmdeploy.serve.openai.api_client import APIClient
import sys
import io
import base64
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *


class ImageDescriptionExtractor:
    SYSTEM_PROMPT = """"""

    USER_PROMPT = """"""

    def __init__(self, api):
        self.api_client = APIClient(api)
        self.model_name = self.api_client.available_models[0]

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

        responds = api_client.chat_completions_v1(model, messages, temperature=0, max_tokens=8296)
        for respond in responds:
            result = respond["choices"][0]['message']['content']
            return result

    def describe(self, ):
        try:
            output = self.call_api(self.api_client, model=self.model_name, prompt={'system': self.SYSTEM_PROMPT, 'user': self.USER_PROMPT})
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


def process_image(augmentor: ImageDescriptionExtractor, figure_txt, fig_root, done_images, output):
    image_name = figure_txt.get('name')
    image_path = figure_txt.get('path')
    if image_name in done_images or not os.path.exists(image_path):
        return

    image = Image.open(image_path)
    response = augmentor.describe()
    if not response:
        return

    match = re.search(r"<[Dd]escription>(.*?)</[Dd]escription>", response, re.DOTALL)
    if match:
        figure_txt['summary'] = match[0]
        save_output(output, figure_txt)
    else:
        print("No find summary")


def save_output(output_file, data):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(api, data, output, fig_root):
    lines = load_data(data)
    done_images = load_done_images(output)
    augmentor = ImageDescriptionExtractor(api)
    print("---------------------------Moedel loading----------------------------")
    for figure_txt in tqdm(lines):
        process_image(augmentor, figure_txt, fig_root, done_images, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=f'http://0.0.0.0:23334', type=str)
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-8B-MPO", type=str)
    parser.add_argument("--data", default="fig_txt_2024-10-01_2024-12-31.jsonl", type=str)
    parser.add_argument("--output", default="test.jsonl", type=str)
    parser.add_argument("--fig-root", default="single_images/", type=str)
    args = parser.parse_args()

    model = args.model
    api = args.api
    data = os.path.join(DATA_OUTPUT_DIR, args.data)
    output = os.path.join(DATA_OUTPUT_DIR, args.output)
    fig_root = os.path.join(DATA_OUTPUT_DIR, args.fig_root)
    print("---------------------------Start----------------------------")
    main(api, data, output, fig_root)
