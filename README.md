# Arknights-Drop-Recognition
This project is a prototype module for image recognition of the game Arknights. It is designed to correctly identify the type and amount of loot items, stage name and other useful information from the game screenshots. Simple OCR (for digits show in this game only) and image classification functions are implemented.

## Requirements
Python >= 3.5 is required. Required libraries are listed in requirements.txt.

## Usage
```
>>> from fullStream import FullStream
>>> data = FullStream('test.png')
>>> data.print()
STAGE: 1-7
STAR: 3
STATUS: ['正常']
LV: 83
EXP: 2522
ITEMS:
{'id': '5001', 'count': '72', 'name': '声望', 'success': True}
{'id': '2001', 'count': '1', 'name': '基础作战记录', 'success': True}
{'id': '30012', 'count': '2', 'name': '固源岩', 'success': True}
{'id': '4001', 'count': '72', 'name': '龙门币', 'success': True}
```
## Notebook
+ [Demo](./notebook/demo.ipynb)

## TODO
+ Access to api of [Penguin-Stats](https://penguin-stats.io)
+ User interface or front-end page
