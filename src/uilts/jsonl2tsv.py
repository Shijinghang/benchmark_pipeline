import json
import re
import base64
import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    json_file = 'public_bench.jsonl'  # 输入的JSON文件
    output_file = 'public_bench.tsv'  # 输出的JSON文件

    with open(json_file, 'r') as file:
        lines = file.readlines()
    datas = []
    for i, data in tqdm(enumerate(lines)):
        data = json.loads(data)
        with open(data['image'], 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        datas.append([
            i + 1,
            image_data,
            data['category'],
            data['subject'],
            data['question'],
            data['A'],
            data['B'],
            data['C'],
            data['D'],
            data['answer'],
        ])

    # Save to TSV
    df = pd.DataFrame(datas,
                      columns=[
                          "index", "image", "category", "l2-category", "question", 'A', 'B',
                          'C', 'D', "answer"
                      ])
    df.to_csv(output_file, sep='\t', index=False)
