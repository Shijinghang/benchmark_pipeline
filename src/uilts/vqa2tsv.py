import json
import pandas as pd
import base64
# 读取json
with open('arxiv_QA.json', encoding='utf-8', mode='r') as f:
    lines = f.readlines()
# Load the JSON data
data = []

for i, line in enumerate(lines):
    json_data = json.loads(line)
    # Extract relevant data
    for conv in json_data['conversations']:
        if conv['from'] == 'user':
            image_path = conv['value'].split('<img>')[1].split('</img>')[0]
            # 将图像转成bs64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
            question = conv['value'].split('<question>')[1].split('</question>')[0]
        elif conv['from'] == 'assistant':
            answer = conv['value'].split('<answer>')[1].split('</answer>')[0]
            data.append([i+1, image_data, image_path, question, "", "", answer, "", "", ""])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["index", "image", "image_path", "question", "hint", "multi-choice options", "answer", "category", "L2-Category", "split"])

# Save to TSV
df.to_csv('arxiv_QA.tsv', sep='\t', index=False)