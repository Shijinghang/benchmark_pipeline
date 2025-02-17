import json
import re
import os
import argparse
from tqdm import tqdm
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *


class TxtAugmenter:
  SYSTEM_PROMPT = """You are a highly skilled research assistant specializing in summarizing image information from **academic papers**. You will receive the image's label, caption, and the paragraph(s) in the paper that mention the image (referred to as the "reference passage"). Additionally, you will be provided with the paper's title and abstract as background information.

Your task is to summarize information related to the target image based on **the provided information**, focusing on extracting key information and improving the clarity and readability of the description. Your result will serve senior scholars, please describe it in a formal and scholarly manner.

**Formatting and Content Restrictions:**
    - Strictly base your summary on the information provided. Do not add personal interpretations or explanations beyond what is explicitly stated in the given text.
    - Ensure all LaTeX formats are deleted, with the exception of mathematical formulas.
    - If the provided content is not related to the target image, ignore it and do not summarize it.
    - When you refer to the target image, use expressions such as "The image" or "The figure," instead of labels such as "Figure~\\ref{?}" or "Figure ?".

**Output Format:**
Please summarize the image information and output the result in XML format with the tag <summary>.
"""

  USER_PROMPT = """
label: {label}
caption: {caption}
refs: {refs}
title: {title}
abstract: {abstract}
"""

  def __init__(self, model, tp=2):
    backend_config = TurbomindEngineConfig(tp=tp)
    self.gen_config = GenerationConfig(temperature=0, max_new_tokens=8192)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def summary(self, label, caption, refs, title, abstract):
    content = self.USER_PROMPT.format(label=label, caption=caption, refs=refs, title=title, abstract=abstract)
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


def process_image(augmentor, figure_txt, fig_root, done_images, output):
  image_label = figure_txt.get('label', '')
  image_path = figure_txt.get('path')
  if image_label in done_images or not os.path.exists(image_path):
    return

  caption = figure_txt.get('caption', '')
  ref = figure_txt.get('ref', [])
  title = figure_txt.get('paper_title', '')
  abstract = figure_txt.get('paper_abstract')
  refs = "\n\n".join(ref)

  response = augmentor.summary(image_label, caption, refs, title, abstract)
  if not response:
    return


#   match = re.search(r"{(.)*['\"]summary['\"]:(.)*}", response, re.DOTALL)
  match = re.search(r"<[Ss]ummary>(.*?)</[Ss]ummary>", response, re.DOTALL)

  #   try:
  #     summary = json.loads(response)['summary']
  #   except Exception as e:
  #     print(e)
  if match:
    figure_txt['summary'] = match[0]
    save_output(output, figure_txt)
  else:
    print("No find summary")


def save_output(output_file, data):
  with open(output_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(model, data, output, fig_root, tp):
  lines = load_data(data)
  done_images = load_done_images(output)
  augmentor = TxtAugmenter(model=model, tp=tp)
  print("---------------------------Moedel loading----------------------------")
  for figure_txt in tqdm(lines):

    process_image(augmentor, figure_txt, fig_root, done_images, output)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
  parser.add_argument("--data", default="fig_txt_2024-10-01_2024-12-31.jsonl", type=str)
  parser.add_argument("--output", default="test.jsonl", type=str)
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
