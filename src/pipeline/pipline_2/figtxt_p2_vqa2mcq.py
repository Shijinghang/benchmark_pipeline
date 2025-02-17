import json
import re
import os
import argparse
from tqdm import tqdm
import pandas as pd
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


class VQA2MCQCreator:
  PROMPT = """Act like an expert who specializes in designing challenging astronomical exam questions.

### Objective:
You will receive an [Question] and a [Ground Truth answer]. Your task is to generate a multiple-choice question with one correct option and three incorrect but plausible options as follows:
1. Use the [Question] as question.
2. Use the [Ground Truth answer] as the correct option.
2. Create three incorrect but plausible options based on related concepts in the same field.

### Output Format:
<question>...</question>
<A>...</A>
<B>...</B>
<C>...</C>
<D>...</D>
<answer>...</answer>

### Example:
Input:
[Question]: What feature is indicated by the strong vorticity minimum in the right panel of the image?
[Ground Truth answer]: A Rossby Wave Instability(RWl) vortex
Output:
<question>What feature is indicated by the strong vorticity minimum in the right panel of the image?</question>
<A>A single-armed spiral</A>
<B>A localized overdensity</B>
<C>A Rossby Wave Instability(RWl) vortex</C>
<D>A region of high eccentricity</D>
<answer>C</answer>

### Input:
[Question]: {question}
[Ground Truth answer]: {answer}
"""

  def __init__(self, model, tp):
    backend_config = TurbomindEngineConfig(tp=tp)
    self.gen_config = GenerationConfig(temperature=0.3, max_new_tokens=2048)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def create_mcq(self, question, answer):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.PROMPT.format(question=question, answer=answer)
                }
            ],
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

def load_data(data):
  df = pd.read_csv(
        data,
        sep='\t',
    )
  lines = df.to_dict(orient='records')
  return lines



def load_done_images(output):
  if os.path.exists(output):
    df = pd.read_csv(
        output,
        sep='\t',
    )
    if df.empty:
      return [], []
    final_questions = df.values.tolist()
    done_images = df['image_path'].tolist()
    return final_questions, done_images
  else:
    return [], []


def process_image(index, creator, question, answer, image_path, explain):

  response = creator.create_mcq(question, answer)
  print(response)
  
  if not response:
    return
  question = QExtract.question_pattern.findall(response)[0]
  answer = QExtract.answer_pattern.findall(response)[0]
  A = QExtract.a_pattern.findall(response)[0]
  B = QExtract.b_pattern.findall(response)[0]
  C = QExtract.c_pattern.findall(response)[0]
  D = QExtract.d_pattern.findall(response)[0]
  return [index, image_path, question, A, B, C, D, answer, explain]


def main(model, data, output, tp):

  lines = load_data(data)
  final_questions, done_images = load_done_images(output)
  creator = VQA2MCQCreator(model=model, tp=tp)

  index = len(final_questions) + 1
  for line in tqdm(lines):
    question = line.get('question', '')
    answer = line.get('answer', '')
    image_path = line.get('image_path', '')
    explain = line.get('explain', '')
    if image_path in done_images:
      continue

    try:
      result = process_image(index, creator, question, answer, image_path, explain)
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
              "index", "image_path", "question","A", "B", "C", "D", "answer", 'explian'
          ],
      )
      df.to_csv(output, sep='\t', index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="mistralai/Mistral-Large-Instruct-2407", type=str)
  parser.add_argument("--data", default="/root/benchmark/data/2024_p2_vqa_test.tsv", type=str)
  parser.add_argument("--output", default="/root/benchmark/pipline/pipline_2/2024_p2_vqa2mcq.tsv", type=str)
  parser.add_argument("--tp", default=2, type=int)

  args = parser.parse_args()

  model = args.model
  data = args.data
  output = args.output
  main(model, data, output, args.tp)
