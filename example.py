from core import FullStream


def print_data(data: FullStream):
    # This function is equal to build-in method FullStream.print()
    print('STAGE:', data.stage)
    print('STAR:', data.star)
    print('STATUS:', data.CODE_STR)
    print('LV:', data.lv)
    print('EXP:', data.exp)
    print('ITEMS:')
    for i in data.item:
        print(i)


if __name__ == '__main__':
    # Scan folder
    # sequence = load_image_path(r'.\TEST')
    # for i in sequence:
    #     data = FullStream(i, debug=True, log=r'.\\debug')
    #     data.print()

    data = FullStream(r'.\\notebook\\test.png')
    print_data(data)
