import cv2
import numpy as np

__all__ = ('binary', 'gray', 'binary_by_color_threshold', 'format_digits_image', 'downsize',
           'find_blob', 'filter_blob', 'drop_error', 'count_lines', 'count_blocks',
           'clean_nesting_rect', 'split_digit',
           'stack_img'
           )


def binary(raw, threshold, positive=1, negative=None):
    """
    Binary an 1-channel image.
    :param raw: 2D image
    :param threshold: Thershold of intensity
    :param positive: Value refers to positive outcome
    :param negative: Value refers to negative outcome
    """
    if negative is None and negative != 0:
        _, result = cv2.threshold(raw, threshold, positive, cv2.THRESH_BINARY)
    else:
        temp = 2 if negative == 1 else 1
        _, result = cv2.threshold(raw, threshold, temp, cv2.THRESH_BINARY)
        result[result == 0] = negative
        result[result == temp] = positive
    return result


def gray(raw):
    """
    Wrap of gray-scale method of openCV.
    """
    return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)


def trim_array_2D(raw, threshold=None, path_only=False, dtype='float32'):
    """
    Trim a 2D-array.
    A threshold is alternative.
    :param raw: 2D-array
    :param threshold: Value
    :param path_only: Return clipped array or clip path
    :param dtype: dtype
    :return: Clipped array by default. If 'path_only' is true, return a clip path.
    """

    raw_copy = raw.astype(dtype)
    v = np.sum(raw_copy, 1)
    if threshold is not None:
        v = np.sign(v - threshold)
    y_begin, y_end = trim_array_1D(v)

    if y_begin < 0:
        return None

    h = np.sum(raw_copy, 0)
    if threshold is not None:
        h = np.sign(h - threshold)
    x_begin, x_end = trim_array_1D(h)

    if path_only:
        return [y_begin, y_end, x_begin, x_end]
    return raw[y_begin:y_end, x_begin:x_end]


def trim_array_1D(raw):
    temp = np.where(raw > 0)[0]
    if len(temp) == 0:
        return (-1, -1)
    return (temp[0], temp[-1] + 1)


def count_lines(raw, threshold):
    """
    count continuous positive blocks in an 1D array
    """
    result = 0
    blocks = []

    temp = np.pad(raw, 1, 'constant')
    temp = temp[1:] - temp[:-1]
    a = np.where(temp > 0)[0]
    b = np.where(temp < 0)[0]

    # if len(a) > len(b):
    #     plot(raw)

    for i in range(len(a)):
        d = b[i] - a[i]
        if d > threshold:
            result += d
            blocks.append([a[i], b[i], d])  # [begin, end, length]

    return result, np.asarray(blocks)


def clean_nesting_rect(rects):
    a = 0
    while a < len(rects):
        for b in range(len(rects)):
            if a != b:
                if rects[b][0] < rects[a][0] < rects[b][0] + rects[b][2] and \
                        rects[b][1] < rects[a][1] < rects[b][1] + rects[b][3]:
                    rects.remove(rects[a])
                    a -= 1
                    break
        a += 1


def binary_by_color_threshold(raw, threshold1=None, threshold2=None, threshold3=None, mode='RGB'):
    """
    Binary image according to the color thresholds in each channel.
    :param raw: 3-channel image
    :param threshold1: Lower threshold
    :param threshold2: Upper threshold
    :param threshold3: Intensity (average of all channels) threshold
    :param mode: Color mode: 'RGB' by default, 'BGR' alternative.
    :return: 1-channel binarized image, where 1 refers to positive, 0 refers to negative.
    """

    H, W = raw.shape[:2]
    result = np.ones((H, W))
    r = raw[:, :, 0]
    g = raw[:, :, 1]
    b = raw[:, :, 2]

    if threshold1 is not None:
        if mode == 'BGR':
            t1 = [threshold1[i] for i in [2, 1, 0]]
        else:
            t1 = threshold1
        result[t1[0] > r] = 0
        result[t1[1] > g] = 0
        result[t1[2] > b] = 0

    if threshold2 is not None:
        if mode == 'BGR':
            t2 = [threshold2[i] for i in [2, 1, 0]]
        else:
            t2 = threshold2
        result[t2[0] < r] = 0
        result[t2[1] < g] = 0
        result[t2[2] < b] = 0

    if threshold3 is not None:
        l = 0.0 + r + g + b
        t3 = threshold3 * 3
        result[l <= t3] = 0

    return result


def drop_error(raw, threshold=10):
    """
    Drop the most deviated elements in the array,
     until remaining elements are ranging within the threshold.
    """

    result = raw.copy()
    while True:
        a = 0
        l = len(result)
        m1 = l - 1
        m2 = 0
        for i in range(l):
            a += result[i]
            if result[m1] < result[i]:
                m1 = i
            if result[m2] > result[i]:
                m2 = i
        a /= l
        if result[m1] - a > threshold or a - result[m2] > threshold:
            if result[m1] - a > a - result[m2]:
                result = np.delete(result, m1)
            else:
                result = np.delete(result, m2)
        else:
            break
    return result


def count_blocks(raw_binary, axis, threshold1, threshold2):
    raw_copy = raw_binary.astype('float32')
    new_sign = np.sign(np.sum(raw_copy, axis) - threshold1 - 0.5) + 1
    _, blocks = count_lines(new_sign, threshold2)
    return blocks


def format_digits_image(digits, H=40, pad=3):
    """
    Rescale and format the extracted digits images.
    Input images should be 1-channel gray-scaled images.
    :param digits: List of extracted digit images: list( dict(x=int, img=White-Black gray image) )
    :param H: Height of output
    :param pad: Padding width
    :return: 1-channel image with white background
    """
    digits.sort(key=lambda x: x['x'])
    temp = []
    padding = np.full((H, pad), 255)
    temp.append(padding)
    for digit in digits:
        digit_image = digit['img']
        digit_image = cv2.resize(digit_image, (int(H * digit_image.shape[1] / digit_image.shape[0]), H))
        temp.append(digit_image)
        temp.append(padding)
    temp = tuple([i for i in temp])
    new_image = np.hstack(temp)
    new_image = np.pad(new_image, ((pad, pad), (0, 0)), mode='constant', constant_values=255)
    return new_image


def find_blob(raw):
    """
    Wrap of cv2.connectedComponentsWithStats.
    """
    raw_copy = np.uint8(raw)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_copy)
    blobs = []
    for i in range(n):
        blobs.append(dict(
            index=i,
            x=stats[i][0],
            y=stats[i][1],
            W=stats[i][2],
            H=stats[i][3],
            size=stats[i][4],
            cx=centroids[i][0],
            cy=centroids[i][1],
        ))
    return labels, blobs


def filter_blob(input, threshold=0.0115, filter=True, W_H=0.625, margin=0.08):
    H, W = input.shape[:2]

    blob_map, old_blobs = find_blob(input)

    size_d = H * W * threshold

    margin_valid = margin * H

    valid_blobs = []
    if filter:
        for blob in old_blobs:
            x0 = blob['x']
            x1 = blob['x'] + blob['W']
            y0 = blob['y']
            y1 = blob['y'] + blob['H']
            size = blob['size']
            cx = blob['cx']
            cy = blob['cy']

            # cut margin
            if margin_valid > y1 or H - margin_valid < y1 or margin_valid > y0 or H - margin_valid < y0:
                continue
            if margin_valid > x1 or W - margin_valid < x1 or margin_valid > x0 or W - margin_valid < x0:
                continue

            y_d = blob['H'] / H
            y_m = cy / H

            if y1 / H > 0.8 or \
                    0.62 < y_m or y_m < 0.29 or \
                    0.25 > y_d or y_d > 0.58:
                continue

            if size < size_d:
                continue

            max_d = max(blob['H'], blob['W'])
            temp = size / blob['H'] / blob['W']
            temp_m = size / (max_d ** 2)

            if blob['W'] / W > 0.15 and temp > 0.75:
                continue

            # this threshold relates to the threshold of binary threshold
            if temp_m > 0.51:
                continue
            valid_blobs.append(blob)
    else:
        valid_blobs = old_blobs

    blobs = [dict(index=0, size=0, x=W, y=H, W=0, H=0)]

    for blob in valid_blobs:
        if blob['index'] == 0:
            continue
        x0 = blob['x']
        x1 = blob['x'] + blob['W']
        y0 = blob['y']
        y1 = blob['y'] + blob['H']

        if blobs[0]['x'] > x0:
            blobs[0]['x'] = x0
        if blobs[0]['y'] > y0:
            blobs[0]['y'] = y0
        if blobs[0]['W'] < x1:
            blobs[0]['W'] = x1
        if blobs[0]['H'] < y1:
            blobs[0]['H'] = y1
        blobs[0]['size'] += blob['size']

        temp = blob.copy()
        temp['count'] = temp['W'] / temp['H'] / W_H
        blobs.append(temp)
    blobs[0]['W'] = blobs[0]['W'] - blobs[0]['x'] + 1
    blobs[0]['H'] = blobs[0]['H'] - blobs[0]['y'] + 1

    return blob_map, blobs


def apply_blob(raw, blob_map, blob):
    H, W = raw.shape[:2]
    if len(raw.shape) == 3:
        temp = []
        for i in range(raw.shape[2]):
            temp_channel = raw[:, :, i]
            temp_result = np.zeros((H, W))
            temp_result[blob_map == blob['index']] = temp_channel[blob_map == blob['index']]
            temp.append(temp_result)
        temp = tuple(i for i in temp)
        result = np.dstack(temp)
    else:
        result = np.zeros((H, W))
        result[blob_map == blob['index']] = raw[blob_map == blob['index']]

    x0 = blob['x']
    x1 = blob['x'] + blob['W']
    y0 = blob['y']
    y1 = blob['y'] + blob['H']

    result = result[y0:y1, x0:x1]
    return result


def vertical_average_of_blob(img, blob_map, blob, threshold=0):  # input gray img
    temp = img.copy()
    temp[blob_map != blob['index']] = 0
    temp[img <= threshold] = 0
    aver = np.sum(temp[
                  blob['y']: blob['H'] + blob['y'],
                  blob['x']: blob['W'] + blob['x']
                  ], 0)

    m = np.max(aver)
    if m > 0:
        aver = aver / m
    return aver


def find_local_minimal(input, start, end, ver=0):
    input_copy = np.asarray(input)[start:end]
    if input_copy.size == 0:
        return None
    result = np.argmin(input_copy) + start

    if (ver > 0):
        k1 = result
        k2 = result
        for j in range(result, start, -1):
            if ((input[j] - input[result]) > ver):
                break
            else:
                k1 = j
        for j in range(result, end):
            if ((input[j] - input[result]) > ver):
                break
            else:
                k2 = j
        return [(k1 + k2) / 2, input[result]]
    return [result, input[result]]


def split_conjunct(img, blob_map, blob, threshold=125):
    aver = vertical_average_of_blob(img, blob_map, blob, threshold)
    count = int(round(blob['count']))
    center = [int((i + 1) * blob['W'] / count) for i in range(count - 1)]
    step = int(blob['W'] / count * 0.3)

    result = []
    for i in center:
        result.append(int(find_local_minimal(aver, i - step, i + step, 0.02)[0]))
    result.append(blob['W'])
    output = [[0, result[0]]]
    for i in range(count - 1):
        output.append([result[i], result[i + 1]])
    return output


def create_digit_img(gray_img, blob_map, blob, seg, threshold=125, positive=0, negative=255, W=8, H=8, rescale=True):
    result = np.full((blob['H'], seg[1] - seg[0]), positive, dtype=np.uint8)

    y1 = blob['y']
    y2 = blob['y'] + blob['H']
    x1 = blob['x'] + seg[0]
    x2 = blob['x'] + seg[1]

    temp = np.ones_like(result)
    temp_img = gray_img[y1: y2, x1: x2]
    temp_blob_map = blob_map[y1: y2, x1: x2]
    temp[temp_img <= threshold] = 0
    temp[temp_blob_map != blob['index']] = 0
    result[temp == 0] = negative

    c = trim_array_2D(temp, 0.6 * H, True)
    if c is None:
        return None

    if not rescale:
        result = result[c[0]:c[1], c[2]:c[3]]
    else:
        result = downsize(result, W, H, x1=c[2], x2=c[3], y1=c[0], y2=c[1])
    return result


def split_digit(gray_img, blob_map, blobs, H=8, W=9, rescale=True, threshold=125, threshold_0=125):
    output = []
    for blob in blobs:
        if blob['index'] == 0:
            continue
        if 'count' in blob and blob['count'] > 1.6:
            temp = split_conjunct(gray_img, blob_map, blob, threshold_0)

            for i in temp:
                temp_img = create_digit_img(gray_img, blob_map, blob, i, threshold=threshold, W=W, H=H, rescale=rescale)
                if temp_img is not None:
                    output.append(dict(x=blob['x'] + i[0], img=temp_img))
        else:
            temp_img = create_digit_img(gray_img, blob_map, blob, [0, blob['W']], threshold=threshold, W=W, H=H,
                                        rescale=rescale)
            if temp_img is not None:
                if 'result' in blob:
                    output.append(dict(x=blob['x'], img=temp_img, result=blob['result']))
                else:
                    output.append(dict(x=blob['x'], img=temp_img))

    return output


def downsize(img, new_W, new_H, x1=None, x2=None, y1=None, y2=None):
    # region: [x1, x2), [y1, y2)
    # size if form of [x, y]
    if x1 is None:
        H, W = img.shape[:2]
    else:
        H = y2 - y1
        W = x2 - x1

    y_0 = 0 if y1 is None else y1
    x_0 = 0 if x1 is None else x1

    result = np.zeros((new_H, new_W))
    dx = W / new_W
    dy = H / new_H
    for i in range(new_H):
        for j in range(new_W):
            y_2 = round((i + 1) * dy) + y_0
            y_1 = round(i * dy) + y_0
            x_2 = round((j + 1) * dx) + x_0
            x_1 = round(j * dx) + x_0
            result[i, j] += np.average(img[y_1:y_2, x_1: x_2])
    return result


def stack_img(input_list, column=10):
    a = len(input_list) // column
    b = len(input_list) % column
    if b > 0:
        a += 1
    # print(a, b)

    if a == 0:
        return None
    if a == 1:
        return np.hstack(
            [np.pad(input_list[j], 1, 'constant', constant_values=255) for j in range(len(input_list))]
        )

    temp = [np.hstack(
        [np.pad(input_list[j], 1, 'constant', constant_values=255) for j in range(i * column, (i + 1) * column)]
    ) for i in range(a - 1)]

    result = np.vstack(temp)

    temp = np.hstack([np.pad(input_list[j], 1, 'constant', constant_values=255)
                      for j in range((a - 1) * column, len(input_list))])
    temp = np.pad(temp, ((0, 0), (0, result.shape[1] - temp.shape[1])), 'constant', constant_values=255)

    result = np.vstack((result, temp))

    return result
