import json
import torch
import re
import os
import transformers
import argparse
from tqdm import tqdm
import pandas as pd
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


class VQACreator:
  SYSTEM = """Act like an expert who specialises in designing challenging astronomical image testing problems that require astronomical expertise and thinking to solve.
### Objective:
You will receive [visual information] of an image and its[associated descriptions].Your task is to generate a clear, challenging and accurate astronomical multiple choice question at a professional level. The question should test the respondent's ability to analyse images and apply comprehensive astronomical knowledge.
### Detailed Instructions:
1. Do not mention the visual information and associated description that was provided to you. The respondent does not have access to those details. The respondent is only provided with questions and images.
2. Avoid hints in the question stem.
3. Ensure the problem must be solved through image analysis, astronomical knowledge, and CoT.
4. Include one correct answer and three plausible but incorrect options, with the correct answer's position randomly assigned.
5. Provide an explanation for the correct answer.

### Output Format:
<question>...</question>
<A>...</A>
<B>...</B>
<C>...</C>
<D>...</D>
<answer>...</answer>
<explain>...</explain>

### Example:
<question>What feature is indicated by the strong vorticity minimum in the right panel of the image?</question>
<A>A single-armed spiral</A>
<B>A localized overdensity</B>
<C>A Rossby Wave Instability(RWl) vortex</C>
<D>A region of high eccentricity</D>
<answer>C</answer>
<explain>The right panel of the image illustrates the disc vorticity, with a strong vorticity minimum highlighted. This minimum corresponds to the presence of a Rossby Wave Instability (RWl) vortex, which is a key feature in the inner regions of the simulated circumbinary disc. </explain>

"""
  PROMPT_MCQ = """
[visual information]: {visual_info}
[associated description]: {description}
Please generate a clear, challenging and accurate astronomical multiple choice question at a professional level.
"""
  def __init__(self, model, tp):
    backend_config = TurbomindEngineConfig(tp=tp)
    self.gen_config = GenerationConfig(temperature=0.2, max_new_tokens=8192)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def create_mcq(self, description, visual_info):
    content = self.PROMPT_MCQ.format(description=description, visual_info=visual_info)
    messages = [
        {
            "role": "system",
            "content": self.SYSTEM
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


class QExtract:
  question_pattern = re.compile(r"<question>(.*?)</question>", re.DOTALL)
  a_pattern = re.compile(r"<A>(.*?)</A>", re.DOTALL)
  b_pattern = re.compile(r"<B>(.*?)</B>", re.DOTALL)
  c_pattern = re.compile(r"<C>(.*?)</C>", re.DOTALL)
  d_pattern = re.compile(r"<D>(.*?)</D>", re.DOTALL)
  answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
  explain_pattern = re.compile(r"<explain>(.*?)</explain>", re.DOTALL)


def load_data(data):
  with open(data, "r", encoding="utf-8") as f:
    return [json.loads(line) for line in f]


def load_done_images(output):
  if os.path.exists(output):
    df = pd.read_csv(output, sep='\t', )
    if df.empty:
      return [], []
    final_questions = df.values.tolist()
    done_images = df['image_path'].tolist()
    return final_questions, done_images
  else:
    return [], []


def process_image(index, creator, image_path, description, visual_info):

  response = creator.create_mcq(description, visual_info)
  if not response:
    return
  question = QExtract.question_pattern.findall(response)[0]
  answer = QExtract.answer_pattern.findall(response)[0]
  explain = QExtract.explain_pattern.findall(response)[0]
  A = QExtract.a_pattern.findall(response)[0]
  B = QExtract.b_pattern.findall(response)[0]
  C = QExtract.c_pattern.findall(response)[0]
  D = QExtract.d_pattern.findall(response)[0]
  return [index, image_path,question, A, B, C, D, answer, explain]


def main(model, data, output, fig_root, tp):

  lines = load_data(data)
  final_questions, done_images = load_done_images(output)
  creator = VQACreator(model=model, tp=tp)

  index = len(final_questions) + 1
  for line in tqdm(lines):
    image_name = line.get('image_name', '')
    description = line.get('description', '')
    visual_info = line.get('visual_info', '')

    image_path = os.path.join(fig_root, image_name)
    if image_path in done_images:
      continue

    try:
      result = process_image(index, creator, image_path, description, visual_info)
      if not result:
        continue
      index = index + 1
      final_questions.append(result)
    except Exception as e:
      print(e)
    finally:
      df = pd.DataFrame(
          final_questions,
          columns=[
              "index", "image_path", "question", "A", "B", "C", "D", "answer",
              'explian'
          ],
      )
      df.to_csv(output, sep='\t', index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
  parser.add_argument("--data", default="/root/benchmark/data/2024_s2_1.jsonl", type=str)
  parser.add_argument("--output", default="/root/benchmark/data/2024_mcq.tsv", type=str)
  parser.add_argument("--fig-root", default="/mnt/nas/public/fig-txt/2024/single_images/new", type=str)
  parser.add_argument("--tp", default=2, type=int)

  args = parser.parse_args()

  model = args.model
  data = args.data
  output = args.output
  fig_root = args.fig_root
  main(model, data, output, fig_root, args.tp)
