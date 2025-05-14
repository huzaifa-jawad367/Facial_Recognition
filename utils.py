import os
import numpy as np
from PIL import Image

def convert_bgr_to_rgb(dir_name):
    images = os.listdir(dir_name)
    paths = [os.path.join(dir_name, image) for image in images]

    for path in paths:
        img = np.array(Image.open(path))[..., ::-1]
        img = Image.fromarray(img)
        img.save(path)
        

convert_bgr_to_rgb("Face Recognition Dataset-1/Asfand")
convert_bgr_to_rgb("Face Recognition Dataset-1/Omer")
convert_bgr_to_rgb("Face Recognition Dataset-1/Saad")
convert_bgr_to_rgb("Face Recognition Dataset-1/Talha")
convert_bgr_to_rgb("Face Recognition Dataset-1/Wasay")

