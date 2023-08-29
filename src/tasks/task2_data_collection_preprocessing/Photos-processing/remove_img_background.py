from rembg import remove
from PIL import Image
import os

for i in os.listdir('/home/Rust'):

    j = i.rsplit('.', maxsplit=1)[0]
    input_path = '/home/Rust/' + i
    output_path = '/home/Rust2//' + j + ".png"

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)



