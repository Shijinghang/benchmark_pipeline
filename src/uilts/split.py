import os 
import json
with open('/root/benchmark/data/2024_s1_visual_2.jsonl', 'r') as f:
    datas = [json.loads(l) for l in f.readlines()]
    ids = [data['id'] for data in datas]
    
    # 分成两个文件
    with open('/root/2024_s1_2.jsonl', 'r') as f1, open('/root/2024_s1_4.jsonl', 'r') as f3:
        s1_1 = [json.loads(l)['id'] for l in f1.readlines()]
        s1_3 = [json.loads(l)['id'] for l in f3.readlines()]
        
        for i, id in enumerate(ids):
            if id in s1_1:
                with open('/root/2024_s1_visual_2.jsonl', 'a') as f10:
                    f10.write(json.dumps(datas[i]) + '\n')
            elif id in s1_3:
                with open('/root/2024_s1_visual_4.jsonl', 'a') as f4:
                    f4.write(json.dumps(datas[i]) + '\n')