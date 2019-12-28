import cv2
import numpy as np

from .DATA import dropItemSample as item_sample
from .DATA import furnSample as furn_sample
from .DATA import sample_size, furn_sample_size
# from .fileHandle import *
from .ssim import Thumbnail_3

# item_sample = load_data('.\\DATA\\dropItemSample.dat')
# furn_sample = load_data('.\\DATA\\furnSample.dat')
# for i in item_sample:
#     i['img'].reset()
# furn_sample['k'].reset()
# save_data(item_sample, '.\\data\\dropItemSample.dat')
# save_data(furn_sample, '.\\data\\furnSample.dat')
for i in item_sample:
    item_sample[i].initial()
furn_sample['k'].initial()
# print('Item sample initialed.')

# dhash_sample_size = 80

item_list = [i for i in item_sample]


def item_match(item_img):
    thumbnail = Thumbnail_3(cv2.resize(item_img, (sample_size, sample_size)))

    result = np.zeros(len(item_list), dtype=np.float64)
    for k in range(len(item_list)):
        sample_img = item_sample[item_list[k]]
        result[k] = thumbnail - sample_img

    target = np.argmax(result)
    target_ssim = result[target]

    result.sort()

    return item_list[target], target_ssim, result[-1] - result[-2]


def is_furniture(item_img):
    test = cv2.resize(item_img, (furn_sample_size, furn_sample_size))
    test = test * furn_sample['w']
    test = Thumbnail_3(test)
    test = test - furn_sample['k']
    return test
