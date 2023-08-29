from PIL import Image
import os 
from os import listdir


for i in os.listdir('/home/Rust'):

# check if the image ends with png
    if (i.endswith(".jpg")):
        print(i)
        
for i in os.listdir('/home/Rust'):
    j = i.rsplit('.', maxsplit=1)[0]
    input_path = r'/home/Rust/' + i
    output_path = r'/home/Rust2//' + j + ".png"

    image = Image.open(input_path)
#image resize
    scale_factor = 2 # you can define the value to increase the pic, scale_factor=2 will double the size
    new_image = (image.size[0] * scale_factor, image.size[1] * scale_factor)
    output = image.resize(new_image)
    output.save(output_path)       
