import os
import time
from hashlib import sha1
from pathlib import Path

import cv2

from .DATA import item_info
from .cutDrop import DropImage
from .digitDetect import which_digit, which_digit_stage
from .fileHandle import make_folder
from .matchItem import item_match, is_furniture

item_info['3003_old'] = item_info['3003'].copy()
item_info['3003_old']['name'] = '赤金(旧)'

__all__ = ('FullStream', 'CODE')


def load_image(x):
    return cv2.imread(x.encode('gbk').decode())


class CODE:
    __CODE_STR = ['正常',
                  '未知物品',
                  '剿灭作战',
                  '首通',
                  '非三星通关',
                  '关卡识别失败',
                  '物品数字异常',
                  '经验值数字异常',
                  '等级数字异常',
                  '作战名称异常'
                  ]

    NORMAL = 0
    UNKNOWN_ITEM = 1
    CAMPAIGN = 2
    FIRST_TIME = 3
    NOT_3_STAR = 4
    RECOGNITION_FAIL = 5
    ITEM_DIGIT_FAIL = 6
    EXP_DIGIT_FAIL = 7
    LV_DIGIT_FAIL = 8
    STAGE_NAME_FAIL = 9

    @classmethod
    def str(cls, i):
        if type(i) is int:
            return cls.__CODE_STR[i]
        if type(i) is list:
            return [cls.__CODE_STR[x] for x in i]
        if type(i) is set:
            return [cls.__CODE_STR[x] for x in i]


def match_digit(digits, t=0.5, handle=which_digit):
    digits.sort(key=lambda x: x['x'])
    result = ''
    confidence = 1
    detail = [handle(i['img']) if 'result' not in i else (i['result'], 1.0) for i in digits]
    for i in detail:
        if i[1] > t:
            if confidence > i[1]:
                confidence = i[1]
            result += i[0]
    return result, confidence, detail


class FullStream:
    def __init__(self, path, save_memory=False, recognition=True, debug=False, log='.\debug'):
        raw = load_image(path)
        self.hash = sha1(cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (500, 500))).hexdigest()
        self.data = DropImage(raw)
        self.path = Path(path)

        self.DEBUG = debug
        self.log = log

        self.detail_report = None
        self.item = None
        self.stage = None
        self.star = None
        self.exp = None
        self.lv = None
        self.CODE = None
        self.CODE_STR = None
        self.__item_list = None
        if recognition:
            self.recognition()

        if not save_memory:
            self.raw = raw
            self.data.reset()
        else:
            self.raw = None

    def recognition(self):
        self.CODE = set()

        self.__recognition()
        self.check_code()
        self.CODE_STR = CODE.str(self.CODE)
        self.__dirty()

    def check_code(self):
        if '4002' in self.__item_list:
            self.CODE.add(CODE.FIRST_TIME)
        if 'unknown' in self.__item_list:
            self.CODE.add(CODE.UNKNOWN_ITEM)
        if self.data.star != 3:
            self.CODE.add(CODE.NOT_3_STAR)
        if '4003' in self.__item_list:
            self.CODE.add(CODE.CAMPAIGN)

        if CODE.STAGE_NAME_FAIL in self.CODE or self.stage == '':
            self.CODE.add(CODE.RECOGNITION_FAIL)

        if len(self.CODE) == 0:
            self.CODE.add(CODE.NORMAL)

    def __dirty(self):
        self.stage = self.stage.replace('0F-', 'OF-')
        if CODE.CAMPAIGN in self.CODE:
            self.stage = '剿灭作战'
        if self.exp == '0' and self.lv == '10' and '5001' not in self.__item_list:
            self.lv = '100'

    def __recognition(self):
        lv = self.parse_lv()
        exp = self.parse_exp()
        stage = self.parse_stage_name()
        item = self.parse_item()

        if lv is None or min([i[1] for i in lv[2]]) <= 0.5:
            self.CODE.add(CODE.LV_DIGIT_FAIL)
        if exp is None or min([i[1] for i in exp[2]]) <= 0.5:
            self.CODE.add(CODE.EXP_DIGIT_FAIL)
        if stage is None or min([i[1] for i in stage[2]]) <= 0.5:
            self.CODE.add(CODE.STAGE_NAME_FAIL)

        self.__item_list = [] if item is None else item[1]

        self.detail_report = dict(
            lv=lv,
            exp=exp,
            stage=stage,
            item=item[0]
        )

        if lv is not None:
            self.lv = lv[0]
        if exp is not None:
            self.exp = exp[0]
        if stage is not None:
            self.stage = stage[0]
        self.star = self.data.star
        if item is not None:
            self.item = [
                dict(
                    id=i['id'] if i['ssim'] > 0.5 else None,
                    count=i['count'],
                    name=i['name'] if i['ssim'] > 0.5 else 'Unknown',
                    success=i['ssim'] > 0.5
                ) for i in item[0]]

    def result(self):
        return dict(lv=self.lv, exp=self.exp, stage=self.stage, star=self.star, item=self.item, status=self.CODE_STR)

    def print(self):
        print('STAGE:', self.stage)
        print('STAR:', self.star)
        print('STATUS:', self.CODE_STR)
        print('LV:', self.lv)
        print('EXP:', self.exp)
        print('ITEMS:')
        for i in self.item:
            print(i)

    def parse_exp(self):
        if self.data.exp_digit is None or len(self.data.exp_digit) == 0:
            return None
        return match_digit(self.data.exp_digit, 0.35)

    def parse_lv(self):
        if self.data.lv_digit is None or len(self.data.lv_digit) == 0:
            return None
        return match_digit(self.data.lv_digit)

    def parse_stage_name(self):
        temp = self.data.stage_name_digit()
        if temp is None or len(temp) == 0:
            return None
        return match_digit(temp, handle=which_digit_stage)

    def parse_item(self):
        if self.data.items is None:
            return None
        result = []
        item_list = []
        for i in self.data.items:
            item_id, item_ssim, item_differ = item_match(i['img'])
            is_furn = 0
            if item_ssim < 0.5:
                is_furn = is_furniture(i['img'])

            if is_furn > 0.90:
                item_list.append('FURN')
                result.append(
                    dict(
                        id='FURN',
                        ssim=is_furn,
                        differ=None,
                        # pass_test=True,
                        name='家具',
                        # type=None,
                        # rarity=None,
                        count=None,
                    )
                )
            else:
                if item_ssim < 0.5:
                    self.CODE.add(CODE.UNKNOWN_ITEM)
                    item_list.append('unknown')

                    if self.DEBUG:
                        filename = "%(time)d.png" % {'time': time.time() * 1000}
                        filepath = os.path.join(self.log, filename)
                        make_folder(self.log)
                        cv2.imwrite(filepath, i['img'])

                else:
                    item_list.append(item_id)

                _item_count = match_digit(i['digit'])
                if _item_count[0] == '' or any([0.3 < i[1] < 0.5 for i in _item_count[2]]):
                    self.CODE.add(CODE.ITEM_DIGIT_FAIL)
                    if item_id not in ['5001', '4001', '4002']:
                        pass

                result.append(
                    dict(
                        id=item_id,
                        ssim=item_ssim,
                        differ=item_differ,
                        # pass_test=item_ssim > 0.5,
                        name=item_id if item_id not in item_info else item_info[item_id]['name'],
                        # type=item_info[item_id]['type'],
                        # rarity=item_info[item_id]['rarity'],
                        count=_item_count[0],
                        count_ssim=_item_count[1],
                        count_detail=_item_count[2],
                    )
                )
        return result, item_list
