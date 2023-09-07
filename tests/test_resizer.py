from PIL import Image
import numpy as np
import cv2
import sys
sys.path.append('./')
from T2IBenchmark.utils import Resizer

img = Image.new("RGB", (128, 128), "white")

center_coordinates = (64, 64)
radius = 48
color = (0, 0, 0)
thickness = 1
np_image = cv2.circle(np.array(img), center_coordinates, radius, color, thickness)
img = Image.fromarray(np_image)
img.show("Original image")

resizer = Resizer("OpenCV", "bicubic", True, (16, 16))
img_resized = resizer(np_image).astype(np.uint8)
img = Image.fromarray(img_resized)
img.show("OpenCV resizer")

resizer = Resizer("PIL", "bicubic", False, (16, 16))
img_resized = resizer(np_image).astype(np.uint8)
img = Image.fromarray(img_resized)
img.show("PIL resizer")
