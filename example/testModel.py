import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import random
import json
import time

from landmark_detector import KeypointDetector
from face_detector import FaceDetector



def draw_on_image(image, keypoints, box):

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    
    ymin, xmin, ymax, xmax = box
    fill = (255, 0, 0, 45)
    outline = 'red'
    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        fill=fill, outline=outline
    )

    for y, x in keypoints:
        draw.ellipse([
            (x - 2.5, y - 2.5),
            (x + 2.5, y + 2.5)
        ], outline='blue')

    return image_copy

def get_crop(image_array):
    
    image_h, image_w, _ = image_array.shape
    box, _ = face_detector(image_array)

    ymin, xmin, ymax, xmax = box[0]
    h, w = ymax - ymin, xmax - xmin
    margin_y, margin_x = h / 6.0, w / 6.0
    ymin, xmin = ymin - 0.5 * margin_y, xmin - 0.5 * margin_x
    ymax, xmax = ymax + 0.5 * margin_y, xmax + 0.5 * margin_x
    ymin, xmin = np.maximum(int(ymin), 0), np.maximum(int(xmin), 0)
    ymax, xmax = np.minimum(int(ymax), image_h), np.minimum(int(xmax), image_w)
    
    crop = image_array[ymin:ymax, xmin:xmax, :]
    crop = cv2.resize(crop, (64, 64))
    return crop, [ymin, xmin, ymax, xmax]


i = random.randint(0, len(metadata) - 1)

# load and preprocess an image
image_array = cv2.imread(metadata[i][0])
image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

crop, box = get_crop(image_array)
Image.fromarray(crop)

DATA_DIR = '/home/gpu2/hdd/dan/CelebA/val/'
MODEL_PATH = 'model.pb'

keypoint_detector = KeypointDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='1')
face_detector = FaceDetector('model-step-240000.pb', visible_device_list='1')


metadata = []
NUM_ANNOTATIONS = 1000
for n in os.listdir(os.path.join(DATA_DIR, 'annotations'))[:NUM_ANNOTATIONS]:
    a = json.load(open(os.path.join(DATA_DIR, 'annotations', n)))
    image_name = a['filename']
    path = os.path.join(DATA_DIR, 'images', image_name)
    box = a['box']
    metadata.append((path, box))


keypoints = keypoint_detector(crop)

ymin, xmin, ymax, xmax = box
h, w = ymax - ymin, xmax - xmin
scaler = np.array([h/64.0, w/64.0])
keypoints = (keypoints*scaler) + box[:2]

draw_on_image(Image.fromarray(image_array), keypoints, box)




times = []
for _ in range(110):
    start = time.perf_counter()
    _ = keypoint_detector(crop)
    times.append(time.perf_counter() - start)
    
times = np.array(times)
times = times[10:]
print(times.mean(), times.std())