# third-party libraries
"""
解析所有s1获取到的论文中的图文信息，先解析tex文件，获取所有图文信息对；然后根据获取到的图文信息对，将对应的图像文件进行转换（如果需要），然后转到统一的位置；
Keyword arguments:
argument -- description
Return: return_description
"""

import requests
import arxiv
import time
import os
import json
import shutil
import tarfile
import zipfile
import rarfile
import gzip
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from bs4 import BeautifulSoup
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# custom libraries
from get_logger import get_logger
from config import *

logger = get_logger(__file__, log_file=os.path.join(LOG_DIR, "s1_get_recent_papers.log"))


def extract_content(soup):
  ol_element = soup.find('ol', class_='breathe-horizontal')
  if ol_element:
    li_elements = ol_element.find_all('li', class_='arxiv-result')
  else:
    li_elements = []

  papers = []
  for li in li_elements:
    published = False
    comments_tags = li.find_all('p', class_='comments is-size-7')
    for comment_tag in comments_tags:
      txt = comment_tag.text.strip().lower()
      if "accepted" in txt or 'journal ref' in txt:
        published = True

    if published:
      paper = {}

      arxiv_id = li.find('p', class_='list-title is-inline-block').find('a')
      paper['paper_id'] = arxiv_id.text.strip().replace("arXiv:", '')

      arxiv_title = li.find('p', class_='title is-5 mathjax')
      paper['title'] = arxiv_title.text.strip()

      paper_abstract = li.find('p', class_='abstract mathjax').find('span',
                                                                    class_='abstract-full has-text-grey-dark mathjax')
      paper['abstract'] = paper_abstract.text.strip().replace('\n        △ Less', '')
      # 获取标签
      tag_elements_link = li.find_all('span', class_='tag is-small is-link tooltip is-tooltip-top')
      for tag_element in tag_elements_link:
        paper['primary-subject'] = tag_element.text.strip()

      paper['secondary-subject'] = []
      tag_elements_grey = li.find_all('span', class_='tag is-small is-grey tooltip is-tooltip-top')
      for tag_element in tag_elements_grey:
        paper['secondary-subject'].append(tag_element.text.strip())
      papers.append(paper)
      time.sleep(2)
  return papers


def get_paper_ids(from_date, to_date, papers_path):
  papers = []
  number, i = 50, 0
  total = 1
  while total > 0:
    url = (f"https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term="
           f"&terms-0-field=title&classification-physics=y&classification-physics_archives=astro-ph"
           f"&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range"
           f"&date-from_date={from_date}&date-to_date={to_date}&start={number*i}")
    try:
      response = requests.get(url)
      response.raise_for_status()
      content = response.text
      if i == 0:
        match = re.search(r"of\s([\d,]+)\sresults", content)
        if match:
          txt = match.group(1).replace(',', '')
          total = int(txt)
          progress_bar = tqdm(total=total)
      else:
        progress_bar.update(number)
      # 解析 content
      soup = BeautifulSoup(response.text, 'html.parser')

      papers += extract_content(soup)

      total -= number
      logger.info(f"Fetched {len(papers)} paper IDs so far.")
    except requests.RequestException as e:
      logger.error(f"Error fetching paper IDs: {e}")
      break
    i += 1
    time.sleep(5)
  earliest_point = int(from_date.split("-")[0][-2:] + from_date.split("-")[1])
  papers = [paper for paper in papers if int(paper['paper_id'].split(".")[0]) >= earliest_point]
  logger.info(f"Filtered to {len(papers)} papers.")

  papers_file = os.path.join(papers_path, f"papers_{from_date.replace('-','')}_{to_date.replace('-','')}.jsonl")
  with open(papers_file, "w") as f:
    for paper in papers:
      f.write(json.dumps(paper) + '\n')
  logger.info(f"Saved paper IDs to {papers_file}.")

  return [paper['paper_id'] for paper in papers]


def download_arxiv(id, save_path):
  url = f"https://arxiv.org/e-print/{id}"
  try:
    response = requests.get(url, stream=True)
    content_disposition = response.headers.get("Content-Disposition")
    filename = content_disposition.split("filename=")[-1].strip('"') if content_disposition else f"{id}.pdf"
    save_path += os.path.splitext(filename)[1] if not filename.endswith(".tar.gz") else ".tar.gz"

    if os.path.exists(save_path):
      return save_path
    with open(save_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
    logger.info(f"Downloaded: {save_path}")
    return save_path
  except Exception as e:
    logger.error(f"Error downloading {id}: {e}")


def untar(filename, save_path):
  try:
    with tarfile.open(filename) as tar:
      tar.extractall(path=save_path)
    logger.info(f"Extracted tar file: {filename} to {save_path}")
  except tarfile.ReadError as e:
    logger.error(f"Error extracting tar file {filename}: {e}")


def unzip(filename, save_path):
  try:
    with zipfile.ZipFile(filename) as zip_file:
      zip_file.extractall(save_path)
    logger.info(f"Unzipped file: {filename} to {save_path}")
  except zipfile.BadZipFile as e:
    logger.error(f"Error unzipping {filename}: {e}")


def unrar(filename, save_path):
  try:
    with rarfile.RarFile(filename) as rar:
      rar.extractall(path=save_path)
    logger.info(f"Unrared file: {filename} to {save_path}")
  except rarfile.RarCannotExec as e:
    logger.error(f"Error unrared {filename}: {e}")


def ungz(filename, save_path):
  try:
    with gzip.open(filename, "rb") as f_in:
      with open(os.path.join(save_path, "main.tex"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Decompressed gzipped file: {filename} to {save_path}")
  except gzip.BadGzipFile as e:
    logger.error(f"Error decompressing {filename}: {e}")


def decompress(path, save_path):
  if path.endswith(".tar.gz"):
    untar(path, save_path)
  elif path.endswith(".gz"):
    ungz(path, save_path)
  elif path.endswith(".zip"):
    unzip(path, save_path)
  elif path.endswith(".rar"):
    unrar(path, save_path)
  elif path.endswith(".pdf"):
    logger.info(f"PDF file {path} does not need extraction.")
  else:
    logger.warning(f"Unsupported extension: {path}")


def process_id(id):
  real_id = id.replace("/", "_") if "/" in id else id
  save_path = os.path.join(papers_raw_path, real_id)
  decompress_path = os.path.join(papers_decompress_path, real_id)

  if os.path.exists(decompress_path):
    logger.info(f"Already decompressed {real_id}. Skipping.")
    return

  if os.path.exists(save_path):
    os.remove(save_path)

  new_save_path = download_arxiv(id, save_path)
  decompress(new_save_path, decompress_path)
  time.sleep(10)


def download_all(ids, num_threads=5):
  with ThreadPoolExecutor(max_workers=num_threads) as executor:
    tqdm(executor.map(process_id, ids), total=len(ids))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--from_date", type=str, default="2024-10-01", required=False)
  parser.add_argument("--to_date", type=str, default="2024-12-31", required=False)
  parser.add_argument("--num_threads", type=int, default=os.cpu_count(), required=False)
  args = parser.parse_args()

  from_date, to_date, num_threads = args.from_date, args.to_date, args.num_threads

  paper_path = os.path.join(DATA_OUTPUT_DIR, "papers")
  os.makedirs(paper_path, exist_ok=True)
  papers_raw_path = os.path.join(DATA_OUTPUT_DIR, "papers_source")
  os.makedirs(papers_raw_path, exist_ok=True)
  papers_decompress_path = os.path.join(DATA_OUTPUT_DIR, "papers_decompress")
  os.makedirs(papers_decompress_path, exist_ok=True)

  paper_ids = get_paper_ids(from_date, to_date, paper_path)
  download_all(paper_ids, num_threads=num_threads)
