from PIL import Image
import os 
from os import listdir

for i in os.listdir('/home/Healthy'):
    j = i.rsplit('.', maxsplit=1)[0]
    input_path = r'/home/Healthy/' + i
    output_path = r'/home/Healthy1//' + j + ".png"

# maximum pixel size
    maxwidth = 1400  #you can define the values for pic width 
    image = Image.open(input_path)
# calculate the width and height of the original img
    width, height = image.size
# calculate the ratio of the img
    ratio = width / height
# calculate new height of the compressed img
    newheight = maxwidth / ratio
# resize original img
    output = image.resize((maxwidth, round(newheight)))
# save output
    output.save(output_path)
