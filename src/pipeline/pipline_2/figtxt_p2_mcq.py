import json
import re
import os
import argparse
from tqdm import tqdm
import pandas as pd
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


class VQACreator:
  PROMPT = """Act like an expert who specializes in designing challenging astronomical exam questions.

### Objective:
You will receive an [image] and its [associated descriptions]. Your ultimate task is to generate a clear, challenging, and accurate multiple-choice question at a professional level that tests the respondent's ability to analyze images and apply comprehensive astronomical knowledge. The problem should be challenging, but the answer should be brief, preferably a word or phrase. Include one correct option and three plausible but incorrect options.

### Detailed Instructions:
1. Image and Description Analysis:
   - View the [image] provided thoroughly, noting any important subjects, features, and text, etc.
   - Read the [associated descriptions] carefully to determine the relationship between the description and the image, and consider the astronomical knowledge involved.

2. Formulate a Challenging Question:
   - Create a question that requires image analysis, astronomical knowledge, and in-depth analysis to solve, ensuring it does not provide hints.

3. Create Answer Choices:
   - Determine a answer to the question as correct option, requiring that the answer be a word or phrase, instead of a long paragraph.
   - Use your astromical knowledge to develop three plausible but incorrect options. 
   
4. Explanation of the Correct Answer:
   - Provide a detailed explanation for why the correct answer is accurate.

### Output Format:
Please return your reply in the following format: 
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

### [associated description]: 
{description}

Please complete the above tasks carefully as required, as testers will feel respected only if you provide challenging questions.
"""

  def __init__(self, model, tp):
    backend_config = TurbomindEngineConfig(tp=tp)
    self.gen_config = GenerationConfig(temperature=0.2, max_new_tokens=4096)
    self.pipeline = pipeline(
        model,
        backend_config=backend_config,
    )

  def create_mcq(self, description, image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.PROMPT.format(description=description)
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


def process_image(index, creator, image_path, description):

  response = creator.create_mcq(description, image_path)
  print(response)
  
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

    image_path = os.path.join(fig_root, image_name)
    if image_path in done_images:
      continue

    try:
      result = process_image(index, creator, image_path, description)
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
              "index", "image_path", "question", "A", "B", "C", "D", "answer", 'explian'
          ],
      )
      df.to_csv(output, sep='\t', index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="OpenGVLab/InternVL2-8B", type=str)
  parser.add_argument("--data", default="/root/benchmark/data/2024_s1_augment_test.jsonl", type=str)
  parser.add_argument("--output", default="/root/benchmark/pipline/pipline_2/2024_p2_mcq.tsv", type=str)
  parser.add_argument("--fig-root", default="/mnt/nas/public/fig-txt/2024/single_images/new", type=str)
  parser.add_argument("--tp", default=2, type=int)

  args = parser.parse_args()

  model = args.model
  data = args.data
  output = args.output
  fig_root = args.fig_root
  main(model, data, output, fig_root, args.tp)
