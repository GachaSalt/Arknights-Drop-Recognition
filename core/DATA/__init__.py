import json
import os
import pickle

import cv2

from core import ssim
from .dropItemPic import itemImage

# import numpy


sample_size = 20
furn_sample_size = 40

__this_dir, __this_filename = os.path.split(__file__)


def __path(filename):
    return os.path.join(__this_dir, filename)


def __load(filepath):
    result = pickle.load(open(filepath, 'rb'))
    for i in result:
        if len(result[i].shape) == 2:
            result[i] = ssim.Thumbnail(result[i])
        else:
            result[i] = ssim.Thumbnail_3(result[i])
    return result


PATH_dropItemSample = __path('dropItemSample.dat')
PATH_furnSample = __path('furnSample.dat')
PATH_expDigitSample = __path('expDigitSample.dat')
PATH_itemDigitSample = __path('itemDigitSample.dat')
PATH_lvDigitSample = __path('lvDigitSample.dat')
PATH_stageDigitSample = __path('stageDigitSample.dat')

dropItemSample = __load(PATH_dropItemSample)

furnSample = result = pickle.load(open(PATH_furnSample, 'rb'))
furnSample['k'] = ssim.Thumbnail_3(furnSample['k'])

expDigitSample = __load(PATH_expDigitSample)
itemDigitSample = __load(PATH_itemDigitSample)
lvDigitSample = __load(PATH_lvDigitSample)
stageDigitSample = __load(PATH_stageDigitSample)

item_info = json.loads(open(__path('item_table.json'), 'r', encoding="UTF-8").read())['items']

for i in itemImage:
    if i not in dropItemSample:
        dropItemSample[i] = ssim.Thumbnail_3(cv2.resize(itemImage[i], (sample_size, sample_size)))
