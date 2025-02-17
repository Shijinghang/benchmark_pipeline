import subprocess
from lmdeploy.serve.openai.api_client import APIClient
import time
import os
import argparse
import threading
import pandas as pd
import re
from signal import SIGTERM, signal, SIGINT, SIGCHLD
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROMPT_JUDGE = """You are a helpful and precise assistant for checking the quality of the answer.
### Task:
Your will given a pair of [Ground Truth] answer and [User Answer] under an Overarching Question. You need to compare this [User Answer] with the [Ground Truth] to determine the correctness of the user's answer. Assign a binary score based on the correctness: 1 for correct and 0 for incorrect.

### Output format:
<score>0 or 1</score>

### Example1:
## Input:
[Overarching Question]: What is the capital of France?
A: Beijing
B: London
C: Berlin
D: Rome
[Ground Truth]: A
[User Answer]: A

<score>1</score>
### Example2:
## Input:
[Overarching Question]: What is the capital of France?
A: Beijing
B: London
C: Berlin
D: Rome
[Ground Truth]: A
[User Answer]: The answer is A.

<score>1</score>

### Example3:
## Input:
[Overarching Question]: What is the capital of France?
A: Beijing
B: London
C: Berlin
D: Rome
[Ground Truth]: C
[User Answer]: B.

<score>0</score>
"""

processes = []


def start_lmdeploy(model, port, cuda_device, tp):
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = cuda_device

  process = subprocess.Popen([
      'lmdeploy',
      'serve',
      'api_server',
      model,
      '--tp',
      str(tp),
      '--server-port',
      str(port),
      '--backend',
      'turbomind',
  ],
                             env=env)
  print(f"Started {model} lmdeploy serve on port {port} with PID: {process.pid}")
  return process


def judge_predictions(judge_api_client, judge_model_name, result_files):
  for result_file in result_files:
    data = pd.read_csv(result_file, sep='\t')
    scores = []
  for idx, row in data.iterrows():
    try:
      if row['prediction'] in ['A', 'B', 'C', 'D']:
        score = int(row['answer'] == row['prediction'])
        print("Use exact-match")
      else:
        messages = [{
            "role": "system",
            "content": PROMPT_JUDGE
        }, {
            "role":
            "user",
            "content":
            f"[Overarching Question]: {row['question']} \n[Ground Truth]: {row['answer']} \n[User Answer]: {row['prediction']}"
        }]
        for item in judge_api_client.chat_completions_v1(model=judge_model_name,
                                                         messages=messages,
                                                         temperature=0.0,
                                                         max_tokens=1024):
          output = item["choices"][0]['message']['content']
          matches = re.findall(r"<score>(.*?)</score>", output, re.DOTALL)
          score = 0 if matches == [] else matches[-1]
      scores.append(int(score))
    except Exception as e:
      print(e)
      scores.append(0)
  data['score'] = scores
  print(sum(scores) / len(scores))
  data.to_csv(result_file, index=False, sep='\t')


def terminate_processes():
  global processes
  for process in processes:
    try:
      os.killpg(os.getpgid(process.pid), SIGTERM)
      process.wait()
    except Exception as e:
      print(f"Error terminating process {process.pid}: {e}")


def signal_handler(sig, frame):
  print("Terminating all processes...")
  # 暂时屏蔽信号处理以防止递归调用
  signal(SIGTERM, signal.SIG_IGN)
  signal(SIGINT, signal.SIG_IGN)
  terminate_processes()
  exit(0)


def reap_child(signum, frame):
  while True:
    try:
      pid, status = os.waitpid(-1, os.WNOHANG)
      if pid == 0:
        break
    except ChildProcessError:
      break


def check_service_ready(port, timeout=600, interval=10):
  """Check if the service is ready by polling the port"""
  start_time = time.time()
  while time.time() - start_time < timeout:
    try:
      response = APIClient(f'http://0.0.0.0:{port}').available_models
      if response:
        return True
    except Exception as e:
      print(f"Waiting for service on port {port} to be ready...")
    time.sleep(interval)
  raise TimeoutError(f"Service on port {port} did not become ready in {timeout} seconds.")


def main(args):
  try:
    signal(SIGTERM, signal_handler)
    signal(SIGINT, signal_handler)
    signal(SIGCHLD, reap_child)

    cuda_device = ','.join(str(x) for x in args.cuda_devices)
    process = start_lmdeploy(args.judge_model, 20033, cuda_device, len(args.cuda_devices))
    processes.append(process)

    check_service_ready(20033)

    judge_api_client = APIClient(f'http://0.0.0.0:20033')
    judge_model_name = judge_api_client.available_models[0]

    # Judge the predictions
    judge_predictions(judge_api_client, judge_model_name, args.result_files)
  except Exception as e:
    print(f"Error: {e}")
  finally:
    terminate_processes()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Judge the predictions made by models.")
  parser.add_argument('--judge-model',
                      type=str,
                      default='internlm/internlm2-chat-20b',
                      help="Judge the answer is right or not")
  parser.add_argument('--cuda-devices', type=str, nargs='+', default=['0', '1'], help="List of CUDA devices")
  parser.add_argument('--result-files', type=str, nargs='+', default=None, help="List of result files to be judged")

  args = parser.parse_args()
  if args.result_files is None:
    args.result_files = glob.glob('./result/*.tsv')
  main(args)
