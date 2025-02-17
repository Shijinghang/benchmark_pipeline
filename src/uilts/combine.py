import json
import os
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--visual-file", default='/root/benchmark/data/2024_s1_visual_1.jsonl', type=str)
  parser.add_argument("--augment-file", default='/root/benchmark/data/2024_s1_augment_1.jsonl', type=str)
  parser.add_argument("--output", default="/root/benchmark/data/2024_s2_1.jsonl", type=str)
  args = parser.parse_args()

  with open(args.visual_file, 'r') as f:
    visual_data = [json.loads(line) for line in f.readlines()]
    visual_data = {visual_data[i]["image_name"]: visual_data[i] for i in range(len(visual_data))}

  with open(args.augment_file, 'r') as f:
    augment_data = [json.loads(line) for line in f.readlines()]
    augment_data = {augment_data[i]["image_name"]: augment_data[i] for i in range(len(augment_data))}

  for i , item in visual_data.items():
    if augment_data.get(item["image_name"], ''):
      with open(args.output, 'a') as f:
        f.write(
            json.dumps({
                "id": item["id"],
                'image_name': item["image_name"],
                "description": augment_data[item["image_name"]]["description"],
                "visual_info": item["visual_info"],
                'title': item["title"],
                "abstract": item["abstract"],
                "caption": item["caption"],
                "ref": item["ref"],
            }, ensure_ascii=False) + "\n")
