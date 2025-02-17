import json 
fig_txt_template ={
    "id": "",
    "name": "",
    "path": "",
    "url": "",
    "size": {
        "width": 0,
        "height": 0
    },
    "description": "",
    "text": "",
    "caption": "",
    "cite": "arxiv",
    "type": "",
    "category": ""
}

with open(r"/mnt/tianwen-tianqing-nas/tianwen/home/sjh/.data/fig_txt_s_20241001-20241231.jsonl", "rb") as f:
    with open(r"/mnt/tianwen-tianqing-nas/tianwen/home/sjh/dataset/img_txt/arxiv.jsonl", "w", encoding="utf-8") as f_out:
        for line in f.readlines():
            line = json.loads(line)
            if line['flg'] != '1' :
                continue
            fig_txt_template['cite'] = "arxiv:"+line['paper_id']
            fig_txt_template['text'] = line['summary']
            fig_txt_template['category'] = line['paper_primary_subject']
            for key in fig_txt_template.keys():
                if key in line.keys():
                    fig_txt_template[key] = line[key]
            f_out.write(json.dumps(fig_txt_template) + "\n")