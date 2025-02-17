import subprocess
from lmdeploy.serve.openai.api_client import APIClient
import time
import os
import argparse
import threading
import pandas as pd
import re
from signal import SIGTERM, signal, SIGINT, SIGCHLD

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROMPT_ANSWER = """You are a rule abiding answerer
### Task:
Your task is to answer a question. If you are unable to determine the correct answer due to missing information, respond "Not Applicable." Otherwise, Answer the question using A or B or C or D.

### Output format:
<answer> A or B or C or D</answer>

### Example:
Question: What is the capital of China?
A: Beijing
B: London
C: Berlin
D: Rome
## Your output:
<answer>A</answer>
"""

processes = []


def start_lmdeploy(model, port, cuda_device):
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = cuda_device

  process = subprocess.Popen([
      'lmdeploy',
      'serve',
      'api_server',
      model,
      '--tp',
      '1',
      '--server-port',
      str(port),
      '--backend',
      'turbomind',
  ],
                             env=env,
                             preexec_fn=os.setsid)
  print(f"Started {model} lmdeploy serve on port {port} with PID: {process.pid}")
  return process


def initialize_models(args):
  global processes
  models = {}
  for i, model in enumerate(args.models):
    port = args.ports[i]
    cuda_device = args.cuda_devices[i]
    process = start_lmdeploy(model, port, cuda_device)
    processes.append(process)
    models[model] = {"port": port}
  return models


def create_api_clients(models):
  for model in models.keys():
    api_client = APIClient(f'http://0.0.0.0:{models[model]["port"]}')
    model_name = api_client.available_models[0]
    models[model].update({"api_client": api_client, "model_name": model_name})
  return models


def model_answer(api_client, model_name, data, result_file):
  output = data.copy()
  for idx, row in data.iterrows():
    messages = [{
        "role": "system",
        "content": PROMPT_ANSWER
    }, {
        "role": "user",
        "content": f"Question:{row['question']}\nA:{row['A']}\nB:{row['B']}\nC:{row['C']}\nD:{row['D']}"
    }]
    for item in api_client.chat_completions_v1(model=model_name, messages=messages, temperature=0.0, max_tokens=2048):
      prediction = item["choices"][0]['message']['content']
      matches = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
      output.loc[idx, 'prediction'] = 'Not Applicable ' if matches == [] else matches[-1]
      print(prediction)
  output.to_csv(result_file, index=False, sep='\t')


def start_model_threads(models, data, input_file_path):
  threads = []
  result_files = []
  for model in models.keys():
    result_file = f"./result/{model.split('/')[-1]}_{os.path.basename(input_file_path)}"
    result_files.append(result_file)
    api_client = models[model]["api_client"]
    model_name = models[model]["model_name"]
    thread = threading.Thread(target=model_answer, args=(api_client, model_name, data, result_file))
    thread.start()
    threads.append(thread)
  return threads, result_files


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
  global processes
  try:
    signal(SIGTERM, signal_handler)
    signal(SIGINT, signal_handler)
    signal(SIGCHLD, reap_child)

    models = initialize_models(args)

    # Ensure services are fully started
    for port in args.ports:
      check_service_ready(port)

    models = create_api_clients(models)
    data = pd.read_csv(args.file_path, sep='\t')

    threads, result_files = start_model_threads(models, data, args.file_path)

    # Wait for all threads to complete
    for thread in threads:
      thread.join()

    terminate_processes()

  finally:
    terminate_processes()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Start multiple lmdeploy services and use them for predictions.")
  parser.add_argument('--file-path', type=str, default='/root/llm_filter.tsv', help="Path to the input file")
  parser.add_argument('--models',
                      type=str,
                      nargs='+',
                      default=['internlm/internlm2-chat-7b', 'meta-llama/Meta-Llama-3.1-8B-Instruct'],
                      help="List of model names to be used")
  parser.add_argument('--ports', type=int, nargs='+', default=[23333, 23334], help="List of ports for each model")
  parser.add_argument('--cuda-devices', type=str, nargs='+', default=['0', '1'], help="List of CUDA devices")

  args = parser.parse_args()

  if len(args.models) != len(args.ports):
    print("The number of models must be equal to the number of ports.")
    exit(1)
  main(args)
