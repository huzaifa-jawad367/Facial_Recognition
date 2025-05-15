import os
import numpy as np
from PIL import Image
import cv2 as cv
import PIL
import matplotlib.pyplot as plt

def convert_bgr_to_rgb(dir_name):
    images = os.listdir(dir_name)
    paths = [os.path.join(dir_name, image) for image in images]

    for path in paths:
        img = np.array(Image.open(path))[..., ::-1]
        img = Image.fromarray(img)
        img.save(path)

def pad_PIL_image1(image1):
    h, w = image1.size
    if h == w:
        return image1
    elif h > w:
        new_w = h
        left = 0
        top = int((new_w - w)/2)
        result = PIL.Image.new(image1.mode, (h, new_w), (0, 0, 0))
        result.paste(image1, (left, top))
        return result
    else:
        new_h = w
        left = int((new_h - h)/2)
        top = 0
        result = PIL.Image.new(image1.mode, (new_h, w), (0, 0, 0))
        result.paste(image1, (left, top))
        return result
        
def pad_cv_image1(image1):
    h, w = image1.shape[0], image1.shape[1]
    if h == w:
        return image1
    elif h > w:
        new_w = h
        left = 0
        top = int((new_w - w)/2)
        bottom = top
        right = left
        result = np.pad(image1, ((0,0),(top,bottom),(0,0)))
        return result
    else:
        new_h = w
        left = int((new_h - h)/2)
        right = left
        top = 0
        result = np.pad(image1, ((left,right),(0,0),(0,0)))
        return result
    
def pad_to_square(img, pad_color=(0,0,0)):
    """
    Pads img (H×W×C) with pad_color so that it becomes square (S×S×C),
    where S = max(H, W). Centers the original image.
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    dh = size - h
    dw = size - w
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    # using OpenCV
    padded = cv.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color
    )
    return padded

def pad_to_square_np(img, pad_color=0):
    h, w = img.shape[:2]
    size = max(h, w)
    dh, dw = size - h, size - w
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    # pad_width for (H, W, C)
    pad_width = ((top, bottom), (left, right), (0, 0))
    return np.pad(img, pad_width, mode='constant', constant_values=pad_color)



if __name__ == "__main__":
    # convert_bgr_to_rgb("Face Recognition Dataset-1/Asfand")
    # convert_bgr_to_rgb("Face Recognition Dataset-1/Omer")
    # convert_bgr_to_rgb("Face Recognition Dataset-1/Saad")
    # convert_bgr_to_rgb("Face Recognition Dataset-1/Talha")
    # convert_bgr_to_rgb("Face Recognition Dataset-1/Wasay")
    pass
