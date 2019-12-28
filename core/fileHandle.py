import os

# from pathlib import Path
# import pickle
__all__ = ('load_image_path',
           # 'save_data', 'load_data',
           'make_folder')

imageFile = ['.jpg', '.png']


def Path(raw):
    return raw


def load_image_path(path):
    _path = Path(path)
    image_file_set = []
    for root, dirs, files in os.walk(_path, topdown=False):
        for name in files:
            suffix = os.path.splitext(name)
            # print(suffix)
            if suffix is None or suffix[1].lower() not in imageFile:
                continue
            image_file_set.append(os.path.join(root, name))
    return image_file_set


# def save_data(data, path):
#     _path = Path(path)
#     pickle.dump(data, open(_path, 'wb'), pickle.HIGHEST_PROTOCOL)
#
#
# def load_data(path):
#     _path = Path(path)
#     return pickle.load(open(_path, 'rb'))


def make_folder(path):
    _path = Path(path)
    if not os.path.exists(_path):
        os.makedirs(_path)
