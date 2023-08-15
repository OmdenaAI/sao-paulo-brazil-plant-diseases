from PIL import Image
import os
from argparse import ArgumentParser

def convert_images_to_rgb(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            image = Image.open(input_path)
            
            # Convert to RGB mode (3 channels)
            rgb_image = image.convert("RGB")
            
            rgb_image.save(output_path)
            print(f"Converted {filename} to RGB and saved in {output_dir}")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("input_dir", help="Input dir path") 
    parser.add_argument("out_dir", help="Output dir path") 
    args = parser.parse_args()
    
    convert_images_to_rgb(args.input_dir, args.out_dir)
