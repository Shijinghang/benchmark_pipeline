import argparse
import pandas as pd
import base64
from pathlib import Path
from PIL import Image, ImageFile
import io
def resize_and_encode_image(image_path, max_side):
    """
    Resizes and encodes an image to base64 format.

    Args:
        image_path (str): Path to the image file.
        max_side (int): Maximum size (in pixels) for the resized image.

    Returns:
        str or None: The base64 encoded image string on success, None on error.
    """

    try:
        # Allow handling of potentially corrupt images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        print(f"Processing {image_path}...")
        with Image.open(image_path) as img:
            # Calculate scaling factor while maintaining aspect ratio
            scale = max_side / max(img.size)
            if scale < 1:  # Resize only if necessary
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert the image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images(input_file, output_file):
    """Process the input TSV file and store the results in an output TSV file."""
    data = pd.read_csv(input_file, sep='\t')

    if 'image_path' not in data.columns:
        raise ValueError("The input file must contain an 'image_path' column.")

    data['image'] = data['image_path'].apply(lambda x: resize_and_encode_image(x, 1024))

    data.to_csv(output_file, sep='\t', index=False)
    print(f"Processed file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Encode images to Base64 and save to TSV after resizing.")
    parser.add_argument("--input-file", type=str, required=True, help="The input TSV file path.")
    parser.add_argument("--output-file", type=str, required=True, help="The output TSV file path.")
    
    args = parser.parse_args()

    process_images(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
