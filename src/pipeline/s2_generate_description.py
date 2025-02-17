import json
import re
import os
import argparse
from tqdm import tqdm
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *


class ImageDescriptionExtractor:
  PROMPT = """Act like a professional visual content analyst with over 20 years of experience in image data analysis. You specialize in interpreting complex charts, images, and graphics, ensuring that every relevant visual detail is accurately captured and thoroughly described.
>>> Task: 
Your task is to provide a comprehensive and detailed description of the visual information presented in the image.

>>> Guidelines: 
1. Please provide a detailed overview of the content of the image in formal and professional language. Your description should be accurate, thorough, and objective, ensuring that all important visual elements are clearly expressed.
2. Your description should be as comprehensive and detailed as possible about each subject, from whole to part.
3. There is no need for extended reasoning, just an objective description of the content in the image.

>>> Output Format:
<visual-information>
your visual description
</visual-information>
"""

  def __init__(self, model, tp):
    backend_config = TurbomindEngineConfig(tp=tp)
    self.gen_config = GenerationConfig(temperature=0.3, max_new_tokens=4096)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def get_visual_info(self, image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.PROMPT_VISUAL
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
            ],
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
    return [json.loads(line) for line in f]


def load_done_images(output):
  if os.path.exists(output):
    with open(output, "r", encoding="utf-8") as f:
      return [json.loads(l)['image_name'] for l in f.readlines()]
  return []


def process_image(image_info_extractor, id, title, abstract, item, fig_root, done_images, output):
  image_name = item.get('name', '')
  image_path = os.path.join(fig_root, image_name)
  if image_name in done_images or not os.path.exists(image_path):
    return
  caption = item.get('caption', '')
  ref = item.get('ref', [])

  response = image_info_extractor.get_visual_info(image_path)

  if not response:
    return

  match = re.search(r"<[Vv]isual-information>(.*?)</[Vv]isual-information>", response, re.DOTALL)
  if match:
    augment_description = match.group(1).strip()
    save_output(
        output, {
            "id": id,
            "title": title,
            "abstract": abstract,
            "image_name": image_name,
            "visual_info": augment_description,
            "caption": caption,
            "ref": ref
        })
  else:
    print(f"Error: No description found in response: {response}")


def save_output(output_file, data):
  with open(output_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(model, data, output, fig_root, tp):
  lines = load_data(data)
  done_images = load_done_images(output)
  print("---------------------------Moedel loading----------------------------")
  image_info_extractor = ImageDescriptionExtractor(model=model, tp=tp)

  for line in tqdm(lines):
    id = line.get('id', '')
    title = line.get('title', '')
    abstract = line.get('abstract', '')
    for item in line.get('data', []):
      process_image(image_info_extractor, id, title, abstract, item, fig_root, done_images, output)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="models--OpenGVLab--InternVL2_5-8B-MPO", type=str)
  parser.add_argument("--data", default="fig_txt_2024-10-01_2024-12-31.jsonl", type=str)
  parser.add_argument("--output", default="2024_s1_visal.jsonl", type=str)
  parser.add_argument("--fig-root", default="single_images/", type=str)
  parser.add_argument("--tp", default=2, type=int)
  args = parser.parse_args()

  model = args.model
  data = os.path.join(DATA_OUTPUT_DIR, args.data)
  output = os.path.join(DATA_OUTPUT_DIR, args.output)
  fig_root = os.path.join(DATA_OUTPUT_DIR, args.fig_root)
  tp = args.tp
  print("---------------------------Start----------------------------")
  main(model, data, output, fig_root, tp)
