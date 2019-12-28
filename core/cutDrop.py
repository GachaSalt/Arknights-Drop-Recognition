import cv2
import numpy as np

from .cvBasical import *

try:
    from .plot import *
except Exception as e:
    print(e)
    print('\033[31mUnable to show plot. \033[0m')
    _PLOT_IMPORTED = False
else:
    _PLOT_IMPORTED = True

__all__ = 'DropImage'


class DropImage:
    def __init__(self, img):
        self.__loaded = False
        if img is not None:
            exp_digit, lv_digit, items, stage_name_img, star, split_point = main(img)
            self.exp_digit = exp_digit
            self.lv_digit = lv_digit
            self.items = items
            self.stage_name_img = stage_name_img
            self.star = star
            self.split_point = split_point
            self.__loaded = True
            self.__stage_name_digit = None
            self.__exp_text_img = None
            self.__lv_text_img = None

    def exp_text_img(self):
        if self.__exp_text_img is None:
            self.__exp_text_img = format_digits_image(self.exp_digit)
        return self.__exp_text_img

    def lv_text_img(self):
        if self.__lv_text_img is None:
            self.__lv_text_img = format_digits_image(self.lv_digit)
        return self.__lv_text_img

    def stage_name_digit(self):
        if self.__stage_name_digit is None:
            self.__stage_name_digit = extract_digit_normal(self.stage_name_img, 1.4)
        return self.__stage_name_digit

    def plot_all(self):
        if not _PLOT_IMPORTED:
            try:
                import matplotlib
            except Exception as e:
                print(e)

            print('\033[31mUnable to show plot. \033[0m')
            return
        if not self.__loaded:
            return
        show(self.exp_text_img(), 'gray')

        show(self.lv_text_img(), 'gray')
        show(self.stage_name_img, 'gray')
        for i in self.items:
            show(i['img'])
            show(format_digits_image(i['digit']), 'gray')

    def reset(self):
        self.__lv_text_img = None
        self.__exp_text_img = None
        self.__stage_name_digit = None


def main(input, margin=5):
    origin = input[margin:-margin, margin:-margin]
    origin_H, origin_W = origin.shape[:2]
    origin_W_H = origin_W / origin_H
    new_H = 720
    new_W = int(720 * origin_W_H)
    raw = cv2.resize(origin, (new_W, new_H))
    gray_img = gray(raw)
    binary_img = binary(gray_img, 120, 2.0)
    binary_img = np.float32(binary_img)

    right_edge, downside_edge, _ = find_important_point(binary_img)
    if right_edge is None:
        return None, None, None, None, None, None

    downside_edge = int(downside_edge / new_H * origin_H)
    right_edge = int(right_edge / new_W * origin_W)

    drop_img = origin[downside_edge:, right_edge:]

    stage_info_img = origin[downside_edge:, 0:right_edge - 10]

    exp_digit, lv_digit, items = parse_drop_img(drop_img)

    if exp_digit is None:
        return None, None, None, None, None, None

    stage_name_img, star = parse_stage_info(stage_info_img)

    return exp_digit, lv_digit, items, stage_name_img, star, [downside_edge, right_edge]


def find_important_point(raw_binary, margin=0):
    horizontal_gradient = np.pad(np.abs(raw_binary[:, 2:] - raw_binary[:, :-2]) / 2, ((0, 0), (1, 1)), 'constant')

    right_edge = 0
    temp_block = []

    vertical_lines_sum = np.sum(horizontal_gradient, 0)

    valid = np.where(vertical_lines_sum > 150)[0]

    max_d = 0
    for i in valid:
        temp_d, temp = count_lines(horizontal_gradient[:, i], 50)
        if max_d < temp_d:
            temp_block = temp[np.argmax(temp[:, 2])]
            max_d = temp_d
            right_edge = i

    if len(temp_block) == 0:
        return None, None, None

    downside_edge = temp_block[0]
    height_of_split_line = temp_block[2]
    return right_edge, downside_edge, height_of_split_line


def parse_drop_img(raw):
    drop_H, drop_W = raw.shape[:2]
    drop_W_H = drop_W / drop_H
    drop_img = cv2.resize(raw, (int(400 * drop_W_H), 400))

    drop_gray = gray(drop_img)
    drop_binary = binary(drop_gray, 62, 1.0)

    drop_binary_upper = drop_binary[:90, :]

    drop_binary_bottom = cv2.dilate(drop_binary[90:, :], cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16)))
    drop_binary_bottom = cv2.erode(drop_binary_bottom, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))

    # find exp bar
    contours, hierarchy = cv2.findContours(drop_binary_upper, 1, 2)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        wh = w / h
        if y < 100 and h > 30 and wh > 9:
            exp_rect = (x, y, x + w, y + h)

    # find items
    item_rect = []

    # show(drop_binary_bottom, 'GRAY')

    contours, hierarchy = cv2.findContours(drop_binary_bottom, 1, 2)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        wh = w / h
        if w > 120 and h > 120 and 0.8 < wh < 1.5 and y > 2:
            item_rect.append([x, y + 90, w, h])

    clean_nesting_rect(item_rect)

    if len(item_rect) == 0:
        return None, None, None

    temp_size = np.asarray(item_rect)[:, 2:4].flatten()
    item_size = int(round(np.average(drop_error(temp_size, 2))))

    exp_img = drop_img[exp_rect[1]:exp_rect[3], exp_rect[0]:exp_rect[2]]
    exp_img = cv2.resize(exp_img, (int(200 * exp_img.shape[1] / exp_img.shape[0]), 200))
    exp_binary = binary_by_color_threshold(exp_img, [90, 90, 90], threshold3=165, mode='BGR')

    # exp_binary = trim_array_2D(exp_binary)

    lv_img = drop_img[0:exp_rect[3] + 5, 10:exp_rect[0] - 1]
    lv_binary = binary_by_color_threshold(lv_img, [0, 50, 100], [90, 180, 255], 85, 'BGR')

    items = []
    for rect in item_rect:
        x1, x2 = rect[0], rect[0] + item_size
        y1, y2 = rect[1], rect[1] + item_size
        digit = extract_digit_item(drop_img[y1:y2, x1:x2])

        items.append(dict(
            img=drop_img[y1:y2, x1:x2],
            digit=digit
        ))

    return extract_digit_normal(exp_binary, 0.71), extract_digit_normal(lv_binary), items


def parse_stage_info(raw):
    img_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    img_binary = binary(img_gray, 230, 1)

    stage_rect = cut_stage_name(img_binary)
    stage_name_img = np.pad(img_binary[stage_rect[2]:stage_rect[3],
                            stage_rect[0]:stage_rect[1]], 3, 'constant')

    star_binary = binary_by_color_threshold(raw[stage_rect[2]:2 * stage_rect[3], stage_rect[1]:],
                                            [0, 200, 200], [150, 255, 255], mode='BGR')
    star = count_blocks(star_binary, 0, 5, 15)

    return stage_name_img, len(star)


def cut_stage_name(raw_binary, margin=5):
    raw_copy = raw_binary.astype('float32')
    vertical_sign = np.sign(np.sum(raw_copy, 1) - 5)

    y_begin = 0
    y_end = 0
    for i in range(margin, len(vertical_sign) - margin):
        if vertical_sign[i] < 0 and y_end > 0:
            break
        elif vertical_sign[i] > 0:
            y_end = i + 1
        else:
            y_begin = i + 1

    horizontal_sign = np.sign(np.sum(raw_copy[y_begin:y_end, :-100], 0) - 5)

    x_begin = 0
    for i in range(len(horizontal_sign)):
        if horizontal_sign[i] > 0:
            x_begin = i
            break

    x_end = x_begin
    for i in range(x_begin, len(horizontal_sign)):
        if horizontal_sign[i] >= 0:
            x_end = i + 1
            continue
        if i - x_end > 50:
            break

    vertical_sign = np.sign(np.sum(raw_copy[:, x_begin:x_end], 1) - 5)

    y_begin = 0
    y_end = 0
    for i in range(margin, len(vertical_sign) - margin):
        if vertical_sign[i] < 0 and y_end > 0:
            break
        elif vertical_sign[i] > 0:
            y_end = i + 1
        else:
            y_begin = i + 1

    return (x_begin, x_end, y_begin, y_end)


def extract_digit_normal(raw, W_H=0.77):
    blob_map, blobs = find_blob(raw)
    blobs = list(filter(lambda x: x['size'] > 20, blobs))
    for i in blobs:
        if i['index'] == 0:
            continue
        count = i['W'] / i['H']
        if 3 < count:
            if i['size'] / (i['W'] - 2) / (i['H'] - 2) > 1:
                i['count'] = 1
                i['result'] = '-'
            else:
                i['count'] = count / W_H
        else:
            i['count'] = count / W_H

    if blob_map is None:
        return None
    digits = split_digit(raw, blob_map, blobs, rescale=False, threshold=0.5, threshold_0=0.5)
    digits.sort(key=lambda x: x['x'])
    return digits


def extract_digit_item(raw):
    # cut the bottom of item icon
    new_img = raw[int(raw.shape[0] * 0.667):, int(raw.shape[1] * 0.4):]
    new_img = gray(new_img)
    new_img = cv2.resize(new_img, (300, int(300 / new_img.shape[1] * new_img.shape[0])), interpolation=cv2.INTER_CUBIC)

    binary_img = binary(new_img, 105, 1)
    blob_map, blobs = filter_blob(binary_img, W_H=0.625)

    if blob_map is None:
        return None
    digits = split_digit(new_img, blob_map, blobs, rescale=False, threshold=135)
    digits.sort(key=lambda x: x['x'])
    return digits
