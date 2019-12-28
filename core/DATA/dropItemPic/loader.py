import os

import cv2

image_type = ['.png', '.jpg']

__this_dir, __this_filename = os.path.split(__file__)

result = {}
for root, dirs, files in os.walk(os.path.join(__this_dir, 'pic')):
    for name in files:
        ext = os.path.splitext(name)
        if ext[1].lower() in image_type:
            result[ext[0]] = cv2.imread(os.path.join(root, name))
