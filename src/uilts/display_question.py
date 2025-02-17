import argparse
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def wrap_text(text, max_width, draw, font):
    words = text.split(' ')
    wrapped_lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + ' '
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word + ' '
    wrapped_lines.append(current_line)
    
    return '\n'.join(wrapped_lines)

def display_question(data, num=None, image_prefix='', image_replacement='', max_length=800):
    image_paths = []
    for idx, row in data.iterrows():
        blank_image = np.ones((1754, 1240, 3), np.uint8) * 255
        pil_image = Image.fromarray(blank_image)
        draw = ImageDraw.Draw(pil_image)
        
        image_path = row['image_path'].replace(image_prefix, image_replacement)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if h > w:
            new_h = max_length
            new_w = int(w * (max_length / h))
        else:
            new_w = max_length
            new_h = int(h * (max_length / w))
        
        resized_img = cv2.resize(img, (new_w, new_h))
        resized_pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        
        start_x = (1240 - new_w) // 2
        start_y = 20
        pil_image.paste(resized_pil_img, (start_x, start_y))
        
        question_text = f"Question:  {row['question']}\n\n"
        if f"{row['A']}" == "nan":
            options_text = ''
        else:
            options_text = f"A:  {row['A']}\nB:  {row['B']}\nC:  {row['C']}\nD:  {row['D']}\n\n"
        answer_text = f"Answer:  {row['answer']}\n\n"
        explain_text = f"Explanation:  {row['explain']}"
        full_text = question_text + options_text + answer_text + explain_text
        
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 25)
        
        wrapped_text = wrap_text(full_text, max_width=1140, draw=draw, font=font)
        
        text_y = start_y + new_h + 30
        for i, line in enumerate(wrapped_text.split('\n')):
            y = text_y + i * 35
            draw.text((50, y), line, font=font, fill=(0, 0, 0))
        
        final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        image_path = f'./tmp/{idx}.png'
        cv2.imwrite(image_path, final_image)
        image_paths.append(image_path)
        
        if num and idx >= num:
            return image_paths
    return image_paths

def images_to_pdf(image_paths, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    for image_path in image_paths:
        c.drawImage(image_path, 0, 0, width, height)
        c.showPage()
    c.save()

def main(args):
    data = pd.read_csv(args.file_path, sep='\t')
    os.makedirs('./tmp', exist_ok=True)
    image_paths = display_question(data, num=args.num_images, image_prefix=args.image_prefix, image_replacement=args.image_replacement, max_length=args.max_length)
    images_to_pdf(image_paths, args.output_pdf)
    
    for image_path in image_paths:
        os.remove(image_path)
    os.rmdir('./tmp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images and PDF from TSV data")
    parser.add_argument('--file_path', type=str, default='/root/2024_p2_mcq_plus_mllm_sample1800.tsv', help="Path to the TSV file")
    parser.add_argument('--num_images', type=int, default=50, help="Number of images to process")
    parser.add_argument('--output_pdf', type=str, default="output_mcq_mllm_smaple1800.pdf", help="Output PDF file path")

    parser.add_argument('--image_prefix', type=str, default='/root/benchmark/data/2024/', help="Prefix of the image path to replace")
    parser.add_argument('--image_replacement', type=str, default='/root/benchmark/data/2024/', help="Replacement for the image path prefix")
    parser.add_argument('--max_length', type=int, default=800, help="Maximum length of the longer side of the image")

    args = parser.parse_args()
    main(args)
