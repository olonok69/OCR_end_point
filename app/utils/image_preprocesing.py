import cv2
from numpy import ndarray, ones, uint8
import pytesseract
import numpy as np


# Some help functions
def get_conf(page_gray):
    """return a average confidence value of OCR result"""
    df = pytesseract.image_to_data(page_gray, output_type="data.frame")
    df.drop(df[df.conf == -1].index.values, inplace=True)
    df.reset_index()
    return df.conf.mean()


def deskew(image):
    """deskew the image"""
    gray = cv2.bitwise_not(image)
    temp_arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(temp_arr > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def resize_img(img_array: ndarray):
    return cv2.resize(img_array, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)


def convert_to_grayscale(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    return img


def denoise_img(img_array):
    kernel = ones((1, 1), uint8)
    img = cv2.dilate(img_array, kernel, iterations=1)
    img = cv2.erode(img_array, kernel, iterations=1)

    return img


def blur_img(img_array):
    bg = cv2.threshold(
        cv2.medianBlur(img_array, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    img = 255 - cv2.absdiff(img_array, bg)

    return img


def get_image_with_bounding_boxes(img_path: str):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(" ")
        img = cv2.rectangle(
            img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2
        )
    return img
