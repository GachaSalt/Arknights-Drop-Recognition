import cv2
import numpy as np

from .DATA import expDigitSample, lvDigitSample, stageDigitSample, itemDigitSample
from .DATA import sample_size


# from .fileHandle import *
from .ssim import Thumbnail


# expDigitSample = load_data(r'.\\DATA\\expDigitSample.dat')
# lvDigitSample = load_data(r'.\\DATA\\lvDigitSample.dat')
# stageDigitSample = load_data(r'.\\DATA\\stageDigitSample.dat')
# itemDigitSample = load_data(r'.\\DATA\\itemDigitSample.dat')


# def __reset_sample(sample, path):
#     for key in sample:
#         sample[key].reset()
#     save_data(sample, path)
#
# __reset_sample(expDigitSample, r'.\\DATA\\expDigitSample.dat')
# __reset_sample(lvDigitSample, r'.\\DATA\\lvDigitSample.dat')
# __reset_sample(stageDigitSample, r'.\\DATA\\stageDigitSample.dat')
# __reset_sample(itemDigitSample, r'.\\DATA\\itemDigitSample.dat')


def __initial_sample(sample):
    for key in sample:
        sample[key].initial()


__initial_sample(expDigitSample)
__initial_sample(lvDigitSample)
__initial_sample(stageDigitSample)
__initial_sample(itemDigitSample)

# print('Digits sample initialed.')


digit_list = [str(i) for i in range(10)]


def __which_digit(img, sample):
    temp = [img - sample[i] for i in digit_list]
    result = np.argmax(temp)
    confidence = temp[result]
    return digit_list[result], confidence


def which_digit(raw, fast=False, t=0.8):
    if type(raw) is not Thumbnail:
        if raw.shape[0] != sample_size or raw.shape[1] != sample_size:
            img = Thumbnail(cv2.resize(raw, (sample_size, sample_size)))
        else:
            img = Thumbnail(raw)
    else:
        img = raw

    a1, s1 = __which_digit(img, expDigitSample)

    if fast and s1 > t:
        return a1, s1

    a2, s2 = __which_digit(img, lvDigitSample)

    if fast and s2 > t:
        return a2, s2

    a3, s3 = __which_digit(img, stageDigitSample)

    if fast and s3 > t:
        return digit_list[a3], s3

    a4, s4 = __which_digit(img, itemDigitSample)

    a = [a1, a2, a3, a4]
    s = [s1, s2, s3, s4]

    result = np.argmax(s)

    return a[result], s[result]


stage_digit_list = list(stageDigitSample.keys())


def which_digit_stage(raw):
    if type(raw) is not Thumbnail:
        if raw.shape[0] != sample_size or raw.shape[1] != sample_size:
            img = Thumbnail(cv2.resize(raw, (sample_size, sample_size)))
        else:
            img = Thumbnail(raw)
    else:
        img = raw

    temp1 = [img - stageDigitSample[i] for i in stage_digit_list]
    a1 = np.argmax(temp1)
    s1 = temp1[a1]
    return stage_digit_list[a1], s1
