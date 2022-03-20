#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2022-03-19 4:57 下午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa
import re
import logging
import json

from pathlib import Path

from collections import defaultdict, OrderedDict

# from itertools import islice
# from pathlib import Path
# from typing import *

# from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y.%m.%d %H:%M:%S',
                    level=logging.INFO)


class AlgorithmReadme:
    """
    Algorithm README 生成

    TODO:
        - 问题分类、精选问题才会加入

    """
    logger = logging.getLogger('AlgorithmReadme')
    ALGORITHMS = 'algorithms'
    PROBLEMS = 'problems'
    NOTES = 'notes'

    main_dir = os.path.join('..', ALGORITHMS)
    problems_dir = os.path.join(main_dir, PROBLEMS)
    notes_dir = os.path.join(main_dir, NOTES)
    algorithm_readme_path = os.path.join(main_dir, 'README.md')
    all_problems_info: dict
    '''{
        'path/to/牛客_0050_中等_链表中的节点每k个一组翻转.md': info,
        ...
    }'''
    tag2topic = json.load(open(os.path.join(main_dir, 'tag2topic.json'), encoding='utf8'))
    topic2problems: dict
    '''{
        '合集-xxx': [问题路径]
    }'''
    src2problems: dict

    RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.S)
    RE_TAG = re.compile(r'Tag: (.*?)\s')
    RE_SEP = re.compile(r'[,，、]')

    AUTO_GENERATED = '<!-- Auto-generated -->'

    def __init__(self):
        """"""
        print('=== AlgorithmReadme Start ===')
        self.load_all_problems()
        self.get_tag2problems()
        self.gen_algorithm_readme()
        self.get_topic_collections()
        print('=== AlgorithmReadme End ===')

    def get_main_content(self, fp):  # noqa
        """"""
        if not os.path.exists(fp):
            return []

        lns = []
        for ln in open(fp, encoding='utf8').read().split('\n'):
            lns.append(ln)
            if ln == self.AUTO_GENERATED:
                break

        return lns

    def gen_algorithm_readme(self):
        """"""
        lns = self.get_main_content(self.algorithm_readme_path)

        # 合集
        lns.append('')
        lns.append('## 合集')
        for topic in self.topic2problems:
            if topic.startswith('合集'):
                lns.append(f'- [{topic}](./{self.NOTES}/{topic}.md)')

        # 细分类型
        lns.append('')
        lns.append('## 细分类型')
        for topic in self.topic2problems:
            if not topic.startswith('合集'):
                lns.append(f'- [{topic}](./{self.NOTES}/{topic}.md)')

        fw = open(self.algorithm_readme_path, 'w', encoding='utf8')
        fw.write('\n'.join(lns))

    def get_topic_collections(self):
        """"""
        for topic, problems in self.topic2problems.items():
            fp = os.path.join(self.notes_dir, f'{topic}.md')
            lns = self.get_main_content(fp)

            if not lns:
                lns.append(f'# {topic}')
                lns.append('')
                lns.append(f'- [Problems List](#problems-list)')
                lns.append('')
                lns.append(self.AUTO_GENERATED)

            lns.append('')
            lns.append(f'## Problems List')
            for p in problems:
                p = Path(p)
                lns.append(f'- [{p.name}]({".." / p.relative_to(self.main_dir)})')

            fw = open(fp, 'w', encoding='utf8')
            fw.write('\n'.join(lns))

    def load_all_problems(self):
        """"""
        tmp = dict()
        for prefix, _, files in os.walk(self.problems_dir):
            for fn in files:
                name, ext = os.path.splitext(fn)
                if ext != '.md' or name.startswith('-') or name.startswith('_'):
                    continue

                fp = Path(os.path.join(prefix, fn))
                txt = open(fp, encoding='utf8').read()
                try:
                    info_ret = self.RE_INFO.search(txt)
                    info = json.loads(info_ret.group(1))
                except:
                    raise ValueError(f'parse info error: {fp}')
                fp = self.try_rename(info, fp)
                self.try_add_title(fp, txt)
                tmp[str(fp)] = info

        self.all_problems_info = tmp

    def try_add_title(self, fp, txt):  # noqa
        """"""
        if txt.startswith('##'):
            return
        lns = txt.split('\n')
        info = fp.name.split('_')
        title = f'## {info[3]}（{info[0]}-{info[1]}, {info[2]}）'
        lns.insert(0, title)
        fw = open(fp, 'w', encoding='utf8')
        fw.write('\n'.join(lns))

    def try_rename(self, info, fp):  # noqa
        """"""
        src, no, dif, name = info['来源'], info['编号'], info['难度'], info['标题']
        fn = f'{src}_{no}_{dif}_{name}.md'
        if fn != fp.name:
            self.logger.info(f'rename {fp.name} to {fn}')
            fp = fp.rename(fp.parent / fn)
            command_ln = f'git add "{fp}"'
            self.logger.info(command_ln)
            os.system(command_ln)
        return fp

    def get_tag2problems(self):
        """"""
        tmp = defaultdict(list)
        for fp, info in self.all_problems_info.items():
            tags = [tag.strip() for tag in info['tags']] + [info['来源']]
            tag2topic = {tag: self.tag2topic[tag.lower()] for tag in tags}
            topics = list(tag2topic.values())
            for topic in topics:
                tmp[topic].append(fp)

        for k, v in tmp.items():
            tmp[k] = sorted(v, key=lambda x: Path(x).name)

        self.topic2problems = dict(sorted(tmp.items()))

    def get_head(self, prefix, fn, info):  # noqa
        """"""
        suffix = '-'.join(prefix.split('/')[-2:])
        src, pid, lv, pn = info['来源'], info['编号'], info['难度'], info['标题']
        head = f'`{src} {pid} {pn} ({lv}, {suffix})`'
        return head

    def get_tag2topic(self):
        notes_dir = self.notes_dir
        file_names = os.listdir(notes_dir)

        tmp = dict()
        for fn in file_names:
            topic, _ = os.path.splitext(fn)
            txt = open(os.path.join(notes_dir, fn), encoding='utf8').read()
            tags = self.RE_SEP.split(self.RE_TAG.search(txt).group(1))
            tmp[topic] = tags

        self.tag2topic = {v.lower(): k for k, vs in tmp.items() for v in vs}


def pipeline():
    """"""
    AlgorithmReadme()


class Test:
    def __init__(self):
        """"""
        doctest.testmod()
        # self.test_AlgorithmReadme()

    def test_AlgorithmReadme(self):  # noqa
        """"""
        AlgorithmReadme()
        # print(json.dumps(ar.tag2topic, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    """"""
    pipeline()
