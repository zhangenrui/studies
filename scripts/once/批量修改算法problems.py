#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2022-01-18 11:38 上午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa
import re
import json

# from collections import defaultdict
# from itertools import islice
from pathlib import Path


# from typing import *

# from tqdm import tqdm


def process_one(fp: Path):
    """"""
    from huaytools.python.custom import NoIndentEncoder
    src, no, dif, name = fp.name.rsplit('.', maxsplit=1)[0].split('_')

    txt = open(fp, encoding='utf8').read()
    gs = re.search('<!--(.*?)-->', txt)
    if not gs:
        print(fp)
        return

    tags = gs.group(1).split(':')[1]
    tags = [t.strip() for t in re.split(r'[,，、]', tags)]
    # print(tags)

    if src == '剑指Offer':
        no = no[1:] + '00'

    js = {'tags': NoIndentEncoder.wrap(tags),
          '来源': src,
          '编号': no,
          '难度': dif,
          '标题': name}

    # fp = fp.rename(f'{src}_{no}_{dif}_{name}.md')
    ret = json.dumps(js, cls=NoIndentEncoder, indent=4, ensure_ascii=False)
    # print(ret)

    txt = re.sub('<!--.*?-->', f'<!--{ret}-->', txt, count=1)
    fw = open(fp, 'w', encoding='utf8')
    fw.write(txt)

    # print(fp.parent)
    fp.rename(os.path.join(str(fp.parent), f'{src}_{no}_{dif}_{name}.md'))


def rename_one(fp: Path):
    """"""
    txt = open(fp, encoding='utf8').read()
    gs = re.search(r'<!--(.*?)-->', txt, flags=re.S)
    if not gs:
        print(fp)
        return
    s = gs.group(1)
    # print(s)
    info = json.loads(s)
    src, no, dif, name = info['来源'], info['编号'], info['难度'], info['标题']
    tags = info['tags']
    fp.rename(os.path.join(str(fp.parent), f'{src}_{no}_{dif}_{name}.md'))


def process(dp):
    """"""
    for dp, _, fns in os.walk(dp):
        for fn in fns:
            if not fn.endswith('md'):
                print(dp, fn)
                continue

            fp = Path(os.path.join(dp, fn))
            # process_one(fp)
            rename_one(fp)

def _test():
    """"""
    doctest.testmod()

    # fp = r'/Users/huayang/workspace/my/studies/algorithms/problems/2022/01/LeetCode_0152_中等_乘积最大子数组.md1'
    # process_one(Path(fp))

    fp = r'/Users/huayang/workspace/my/studies/algorithms/problems/2021/11/剑指Offer_1400_中等_1-剪绳子（整数拆分）.md'
    rename_one(Path(fp))


def main():
    """"""
    dp = r'/Users/huayang/workspace/my/studies/algorithms/problems'
    process(dp)


if __name__ == '__main__':
    """"""
    # _test()
    main()
