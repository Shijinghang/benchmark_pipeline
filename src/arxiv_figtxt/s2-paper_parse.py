# third-party libraries
import glob
import shutil
from tqdm import tqdm
import os
import os.path as osp
import re
from PIL import Image, ImageChops
import langdetect
import jsonlines
import pandas as pd
import json
import numpy as np
import uuid
import subprocess
import fitz
import time
import arxiv as arxiv
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import sys
from bs4 import BeautifulSoup
import requests

# custom libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
"""
TODO: mutli图转单图。
"""


def extract_arxiv_category(html_text):
  # 正则表达式模式，用于匹配 arXiv ID
  soup = BeautifulSoup(html_text, 'html.parser')
  primary_subject = soup.find('span', class_='primary-subject').text
  return primary_subject


def get_category(id):
  url = f'https://arxiv.org/abs/{id}'

  response = requests.get(url)
  response.raise_for_status()
  time.sleep(1)
  category = extract_arxiv_category(response.content.decode())
  return category


def pdf2img(pdf_path, tmp_folder=None, scale=1):

  os.makedirs(tmp_folder, exist_ok=True)

  pdfDoc = fitz.open(pdf_path)

  num_page = pdfDoc.page_count
  images = []
  for i in range(num_page):
    page = pdfDoc[i]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    basename = os.path.basename(pdf_path)
    save_path = f"{tmp_folder}/{basename}_{i + 1}.jpg"
    pix.save(save_path)
    images.append(save_path)
  if len(images) == 1:
    return images[0]
  elif len(images) < 1:
    return ""
  else:
    # 合并images中的所有图像成一个图像
    print("mutil-paper")


def eps_to_pdf(eps_path, tmp_folder, pass_exist=False):

  jpg_name = os.path.split(eps_path)[-1].replace(".eps", "jpg")
  jpg_path = os.path.join(tmp_folder, jpg_name)

  # cmd = 'gswin64c -dQUIET -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dNoOutputFonts=false -sPAPERSIZE=a2 -sOutputFile="{}" "{}"'.format(
  #     jpg_path, tmp_folder
  # )
  cmd = "gs -sDEVICE=jpeg -dJPEGQ=95 -r300 -o {} {}".format(jpg_path, eps_path)
  os.system(cmd)
  return jpg_path


def sub_loop(pattern, new, s):
  while True:
    old_s = s
    s = re.sub(pattern, new, s)
    if s == old_s:
      break
  return s


def parse_arxiv_module(id, tex_path, res):
  # print(tex_path)
  try:
    with open(tex_path, "r", encoding="utf-8") as file:
      content = file.read()
  except:
    with open(tex_path, "r", encoding="iso-8859-1") as file:
      content = file.read()
  content = sub_loop(r"(?<!\\)%.*\n", "\n", content)
  if not "\\begin{document}" in content:
    return res

  # 把table去掉
  table_pattern = r"(\\begin\{table\*?\}(.*?)\\end\{table\*?\})"
  table_matches = re.findall(table_pattern, content, re.DOTALL)  # re.DOTALL 取最外面的{}内容
  for table_match in table_matches:
    content = content.replace(table_match[0], "")

  paras = content.split("\n\n")
  # content = content.replace('\n', ' ').replace('(cf. ', '(').replace('Fig. ', 'Fig ').replace('Figs. ', 'Figs ')

  figure_pattern = r"(\\begin\{figure\*?\}(.*?)\\end\{figure\*?\})"
  figure_matches = re.findall(figure_pattern, content, re.DOTALL)  # re.DOTALL 取最外面的{}内容
  for figure_match in figure_matches:
    figure_match = figure_match[0]
    fig_res = {}
    refs = []
    label_pattern = r"\\label\{([^}]+)\}"
    label_match = re.findall(label_pattern, figure_match)
    if len(label_match) > 0:
      label_match = label_match[0]
      for para in paras:
        if ("{" + label_match + "}" in para and "\\label{" + label_match + "}" not in para):
          refs.append(para)
    else:
      continue

    figure_match = figure_match.replace("  ", "")
    fig_res["source"] = figure_match

    image_pattern = (
        r"\\includegraphics *(\[([^]]+)\])? *\{([^}]+)\}"  # [^{}]+ 取最里面的{}内容
    )
    image_matches = re.findall(image_pattern, figure_match)
    image_dict = {}
    for match in image_matches:
      image_dict[match[2]] = match[1]
    fig_res["image"] = image_dict

    k = 0
    captions = []
    while "caption" in figure_match and k < 4:
      k += 1
      caption = find_latex(figure_match, "\\caption")
      captions.append(caption)
      figure_match = figure_match.replace(caption, "")
    caption = "\n\n".join(captions)

    fig_res["caption"] = caption
    fig_res["label"] = label_match
    fig_res["ref"] = refs
    filename = re.sub(r"[^a-zA-Z0-9]", "", label_match)
    name = f"{id}_{filename}.jpg"
    fig_res["name"] = name
    res.append(fig_res)
  return res


def parse_arxiv(folder):
  res = []
  id = osp.split(folder)[1]
  tex_paths = glob.glob(folder + "/*.tex")
  if len(tex_paths) == 0:
    print(folder, "no tex found!")
  else:
    if len(tex_paths) > 1:
      tex_sizes = [os.path.getsize(path) for path in tex_paths]
      tex_paths = np.array(tex_paths)[np.argsort(tex_sizes)[::-1]]
    for tex_path in tex_paths:
      res = parse_arxiv_module(id, tex_path, res)
      if len(res) > 0:
        break
  return res


def filter_paper(key):
  json_path = "E:/dataset/parsing/arxiv/arxiv-metadata-oai-snapshot.jsonl"
  with jsonlines.open(json_path, mode="r") as reader:
    with open(id_path, "w") as f:
      for row in tqdm(reader):
        id = row["id"]
        category = row["categories"]
        title = row["title"]
        if key in category and langdetect.detect(title) == "en":
          f.write(f"{id}\n")


def parse_all_arxiv(paper_folder, figure_info_folder, papers_df: pd.DataFrame):
  progress_bar = tqdm(total=len(papers_df))
  for i, item in papers_df.iterrows():
    id = item["paper_id"]
    title = item['title']
    abstract = item['abstract']
    secondary_subject = item['secondary-subject']
    primary_subject = item['primary-subject']
    figure_info_path = f"{figure_info_folder}/{id}.jsonl"
    if osp.exists(figure_info_path):
      continue
    paper_path = f"{paper_folder}/{id}"
    res = parse_arxiv(paper_path)
    if res:
      with jsonlines.open(figure_info_path, mode="w") as writer:
        data = {
            "paper_id": id,
            "paper_title": title,
            "paper_abstract": abstract,
            "paper_primary_subject": primary_subject,
            "paper_secondary_subject": secondary_subject,
            "data": res
        }
        writer.write(data)
    progress_bar.update(1)


def rm_white_area(tmp_path):
  img = Image.open(tmp_path)
  bg = Image.new(img.mode, img.size, (255, 255, 255))
  diff = ImageChops.difference(img, bg)
  try:
    x1, y1, x2, y2 = diff.getbbox()
    margin = 10
    w, h = img.size
    img = img.crop((
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w, x2 + margin),
        min(h, y2 + margin),
    ))
    img.save(tmp_path)
  except Exception as e:
    print(e)


def find_real_path(id_folder, img_path):
  img_path = img_path.replace(" ", "")
  if "{" in img_path:
    filename = re.findall(r".*\{(.*)\}?", img_path)[0]
  else:
    filename = img_path
  real_path = f"{id_folder}/{filename}"

  if not osp.exists(real_path):
    tmp_paths = (glob.glob(f"{id_folder}/../{filename}") + glob.glob(f"{id_folder}/../{filename}.*") +
                 glob.glob(f"{id_folder}/{filename}.*") + glob.glob(f"{id_folder}/*/{filename}.*") +
                 glob.glob(f"{id_folder}/*/{filename}") + glob.glob(f"{id_folder}/*/*/{filename}.*") +
                 glob.glob(f"{id_folder}/*/*/{filename}"))
    if len(tmp_paths) > 0:
      real_path = tmp_paths[0]

  return real_path


def find_latex(latex, s):
  match = ""
  if s in latex:
    start_idx = latex.index(s)
    end_idx = start_idx + len(s)
    count_left = 1
    count_right = 0
    while count_left != count_right and "}" in latex[end_idx + 1:]:
      end_idx = latex[end_idx + 1:].index("}") + end_idx + 1
      match = latex[start_idx:end_idx + 1]
      count_left = match.count("{")
      count_right = match.count("}")
  return match


def render_from_source(ext, raw_img_path, final_img_path, tmp_folder):

  try:
    if ext == ".pdf":
      # print(f'pdf: {raw_img_path}')
      img_path = pdf2img(raw_img_path, tmp_folder, scale=3)
    elif ext in [".eps", '.ps', '']:
      # print(f'ghostscript: {raw_img_path}')
      img_path = eps_to_pdf(raw_img_path, tmp_folder)
    else:
      print("img:", raw_img_path)
    if not osp.exists(img_path):
      print("img not exist:", raw_img_path)
      return False

    rm_white_area(img_path)
    if osp.exists(final_img_path):
      os.remove(final_img_path)
    shutil.copy(img_path, final_img_path)
    return True
  except Exception as e:
    print(raw_img_path)
    print("error", e)


def compile_tex_to_pdf(tex_file):
  # 获取绝对路径
  tex_file_abs = os.path.abspath(tex_file)
  tex_dir = os.path.dirname(tex_file_abs)
  # 保存当前工作目录
  original_cwd = os.getcwd()
  # 切换到tex文件所在目录
  os.chdir(tex_dir)

  # 确保tex文件存在
  if not os.path.isfile(tex_file_abs):
    print(f"Error: TeX file {tex_file_abs} not found.")
    return

  process = subprocess.Popen(
      ["pdflatex", "-interaction=nonstopmode", "-shell-escape", tex_file_abs],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )

  try:
    # 等待指定时间
    stdout, stderr = process.communicate(timeout=30)
  except subprocess.TimeoutExpired:
    # 超时处理
    process.kill()
    stdout, stderr = process.communicate()
    print("*" * 100)
    print("timeout")
    print(stdout.decode(), stderr.decode())

  # # 执行编译命令
  # result = subprocess.run(
  #     ['pdflatex', '-interaction=nonstopmode', '-shell-escape', tex_file_abs],
  #     capture_output=True, check=True, text=True, encoding='latin-1', timeout=60
  # )

  except Exception as e:
    print(f"Error occurred while compiling {tex_file_abs}:\n{e}")

  finally:
    # 切换回原来的工作目录
    os.chdir(original_cwd)


def process_fig_txt(
    fig_txt,
    check_id,
    run_multi,
    paper_folder,
    multi_image_folder,
    single_image_folder,
    tmp_folder,
):
  key = fig_txt["paper_id"]

  if check_id is not None and key != check_id:
    return
  if len(fig_txt["data"]) == 0:
    return

  try:
    paper_path = os.path.join(paper_folder, key)
    if run_multi:
      save_folder = os.path.join(multi_image_folder, key)
    else:
      save_folder = os.path.join(single_image_folder, key)
    if not osp.exists(save_folder):
      os.makedirs(save_folder)

    for k, info in enumerate(fig_txt["data"]):
      if len(info["ref"]) == 0:
        continue

      name = osp.splitext(info["name"])[0]
      final_img_path = f"{save_folder}/{name}.jpg"

      if check_id is None and osp.exists(final_img_path):
        continue

      source = info["source"].replace("\\plotone", "\\includegraphics")
      image_matches = re.findall(r"(\\includegraphics *(\[([^]]+)\])? *\{([^}]+)\})", source) + re.findall(
          r"(\\plotfiddle *\{([^}]+)\})", source)
      if len(image_matches) == 0:
        continue
      elif len(image_matches) == 1:
        if run_multi:
          continue
        image_filename = list(info["image"].keys())[0]
        raw_img_path = find_real_path(paper_path, image_filename)

        if osp.exists(raw_img_path):
          basename, ext = osp.splitext(raw_img_path)

          if ext in [".pdf", ".ps", ".eps", ""]:
            flag = render_from_source(ext, raw_img_path, final_img_path, tmp_folder)
            if not flag:
              continue
          else:
            img = Image.open(raw_img_path).convert("RGB")
            match = image_matches[0][0]
            if "angle=" in match:
              angle = re.findall(r"angle=(.*),|]", match)[0]
              img = img.rotate(int(angle))
            img.save(final_img_path)
            rm_white_area(final_img_path)
        else:
          print(f"{raw_img_path} is not found!")
      else:
        # print("multi-images", image_matches)
        if run_multi:
          flag = render_from_source(source, name, paper_path, final_img_path)
          if not flag:
            continue
        else:
          continue

  except Exception as e:
    print(e)


def render_figure_texlive(figure_info_folder, ids, tmp_folder, check_id=None, run_multi=False):

  for id in ids:
    fig_txt = json.load(open(f"{figure_info_folder}/{id}.jsonl", "r"))
    # Pool of workers
    process_fig_txt(
        fig_txt,
        check_id,
        run_multi,
        paper_folder,
        multi_image_folder,
        single_image_folder,
        tmp_folder,
    )
    # with Pool(processes=4) as pool:
    #     pool.starmap(
    #         process_row,
    #         [
    #             (
    #                 row,
    #                 check_id,
    #                 run_multi,
    #                 paper_folder,
    #                 multi_image_folder,
    #                 single_image_folder,
    #             )
    #             for row in reader
    #         ],
    #     )

  # if run_multi:
  #     folder = f"{multi_image_folder}/{top_id}"
  # else:
  #     folder = f"{single_image_folder}/{top_id}"
  postprocess_img(folder)


def postprocess_img(folder):
  thres = 200
  top_id = osp.split(folder)[1]
  for path in tqdm(glob.glob(f"{folder}/*")):
    try:
      img = np.array(Image.open(path))
      size = img.shape
      if np.min(img) == 255:
        os.remove(path)
        print(f"empty: {path}\n")
      elif not (np.min(img[0, :]) > thres and np.min(img[-1, :]) > thres and np.min(img[:, 0]) > thres
                and np.min(img[:, -1]) > thres):
        os.remove(path)
        print(f"crop: {path}\n")
      elif size[0] < 100 or size[1] < 100:
        os.remove(path)
        print(f"small: {path}\n")
      if osp.exists(path) and rm_edge_text(path):
        print(f"text: {path}\n")
    except Exception as e:
      print(path, e)


def rm_edge_text(path):
  size = 100
  ori_img = Image.open(path).convert("RGB")
  img = np.array(ori_img)
  img[-size:, :size] = 255
  img = Image.fromarray(img)

  bg = Image.new(img.mode, img.size, (255, 255, 255))
  diff = ImageChops.difference(img, bg)
  x1, y1, x2, y2 = diff.getbbox()
  margin = 10
  w, h = img.size
  if x1 - margin > 0 and y2 + margin < h:
    out_img = ori_img.crop((
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w, x2 + margin),
        min(h, y2 + margin),
    ))
    np_out_img = np.array(out_img)
    np_out_img[:, 0] = 255
    np_out_img[-1, :] = 255
    out_img = Image.fromarray(np_out_img)
    # out_img.show()
    out_img.save(path)
    return True
  else:
    return False


if __name__ == "__main__":
  # 转图片
  paper_folder = f"{DATA_OUTPUT_DIR}/papers_decompress"  # 解压后的原文件路径
  figure_info_folder = f"{DATA_OUTPUT_DIR}/figure_info_1"  # 解压后的原文件路径
  single_image_folder = f"{DATA_OUTPUT_DIR}/single_images"  # 图片保存路径
  multi_image_folder = f"{DATA_OUTPUT_DIR}/multi_images"  # 图片保存路径
  tmp_folder = f"{DATA_OUTPUT_DIR}/tmp"  # 临时文件路径
  log_folder = f"{DATA_OUTPUT_DIR}/logs"

  # 验证所有的folder都存在
  for folder in [
      paper_folder,
      figure_info_folder,
      single_image_folder,
      multi_image_folder,
      tmp_folder,
      log_folder,
  ]:
    if not osp.exists(folder):
      os.makedirs(folder)

  papers_path = f"{DATA_OUTPUT_DIR}/papers/papers_20241001_20241231.jsonl"

  # 创建一个空的 fig_txt.jsonl 文件
  fig_txt_file_name = f"fig_txt_{osp.split(papers_path)[1].removeprefix('papers_')}"
  fig_txt_path = os.path.join(DATA_OUTPUT_DIR, f"{fig_txt_file_name}")
  if not osp.exists(fig_txt_path):
    with open(fig_txt_path, "w") as f:
      pass

  # 获取所有id
  papers_df = pd.DataFrame([json.loads(line) for line in open(papers_path).readlines()])
  papers_ids = set(papers_df['paper_id'].tolist())
  # 获取需要解析的id
  parsed_ids = set([osp.split(path)[1] for path in glob.glob(f"{figure_info_folder}/*")])
  to_parse_ids = list(papers_ids - parsed_ids)

  papers_df = papers_df[papers_df['paper_id'].isin(to_parse_ids)]
  # 解析LaTeX文件并提取图表信息
  parse_all_arxiv(paper_folder, figure_info_folder, papers_df)

  # 渲染图表
  # render_figure_texlive(figure_info_folder, papers_ids, tmp_folder)

  # # 后处理图像
  # folder = f'{single_image_folder}/2024'  # 按照生成图像的路径修改
  # postprocess_img(folder)

  # # 更新和添加信息（可选）
  # update_info()
  # add_info()
  # astro-ph_0104232_f04b161d-3380-11ef-83bb-88d7f63ce820.jpg
