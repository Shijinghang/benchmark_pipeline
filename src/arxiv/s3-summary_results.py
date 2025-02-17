"""sumary_line
  汇总所有文章获取到的图片信息，统一保留在一个jsonl文件中；
  注意：在合并过程中检测图像是否存在或存在问题（有些图像可能在转换过程中出错，
	   前面获取到的每个文章的图像信息要比解析过程中全（比如如果只关注单图像图文的话，s2进会处理单图像，但是s1是获取所有的）所以会有很多不存在的image，如果不存在则跳过；
Keyword arguments:
argument -- description
Return: return_description
"""

import json
import os
from tqdm import tqdm
import glob
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *


def combine_jsonl_files():

  files = glob.glob(os.path.join(DATA_OUTPUT_DIR, "figure_info/*.jsonl"))  # Input JSON file
  figure_info_all = []
  for file in files:
    json_file = file
    figure_info_all.append(json.loads(open(json_file, 'r').readlines()[0]))
  print(len(figure_info_all))
  num = 0
  # Process each line
  results = []
  for figure_info in tqdm(figure_info_all):

    paper_id = figure_info.get('paper_id')
    paper_title = figure_info.get('paper_title')
    paper_abstract = figure_info.get('paper_abstract')
    paper_secondary_subject = figure_info.get('paper_secondary_subject')
    paper_primary_subject = figure_info.get('paper_primary_subject')
    data = figure_info.get('data')
    # Check if the image paths exist and append valid sources to 'datas'
    for source in data:
      image_path = os.path.join(os.path.join(DATA_OUTPUT_DIR, 'single_images/', paper_id, source['name']))
    if os.path.exists(image_path):
      try:
        size = cv2.imread(image_path).shape[:2]
        size = {"width": size[1], "height": size[0]}
      except Exception as e:
        print(e)
        os.remove(image_path)
        continue
      temp_source = {
          "name": source['name'],
          "path": image_path,
          "size": size,
          "label": source['label'],
          'caption': source['caption'],
          "ref": source['ref'],
          "paper_id": paper_id,
          "paper_title": paper_title,
          "paper_abstract": paper_abstract,
          "paper_primary_subject": paper_primary_subject,
          "paper_secondary_subject": paper_secondary_subject,
      }
      results.append(temp_source)
      num += 1
    else:
      print(f"Path does not exist: {image_path}")
  print(num)

  output_file = os.path.join(DATA_OUTPUT_DIR, 'fig_txt_20241001-20241231')
  with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
      f.write(json.dumps(result, ensure_ascii=False) + '\n')


# 检测重复
def check_duplicate():
  with open(os.path.join(DATA_OUTPUT_DIR, 'fig_txt_20241001-20241231'), 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

  # 检测重复
  duplicates = {}
  results = []
  for item in data:
    key = (item['paper_id'], item['name'])
    if key in duplicates:
      continue
    else:
      duplicates[key] = True
      results.append(item)

  print(len(results))
  with open(os.path.join(DATA_OUTPUT_DIR, 'fig_txt_20241001-20241231'), 'w', encoding='utf-8') as f:
    for result in results:
      f.write(json.dumps(result, ensure_ascii=False) + '\n')


def filter_low_quality():
  with open(os.path.join(DATA_OUTPUT_DIR, 'fig_txt_20241001-20241231'), 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

  n = 0
  lens = []
  for item in data:
    caption_len = len(item['caption'].split(' '))
    ref_len = sum([len(i.split(' ')) for i in item['ref']])
    lens.append(ref_len + caption_len)
    if (caption_len) < 90 and ref_len < 90 or (caption_len + ref_len) > 5000:
      n += 1
      item['flg'] = '0'
    else:
      item['flg'] = '1'
  plt.hist(sorted(lens)[:11000], bins=200)
  plt.savefig('hist.png')
  print(n)
  with open(os.path.join(DATA_OUTPUT_DIR, 'fig_txt_20241001-20241231'), 'w', encoding='utf-8') as f:
    for result in data:
      f.write(json.dumps(result, ensure_ascii=False) + '\n')


# combine_jsonl_files()
# check_duplicate()
filter_low_quality()
