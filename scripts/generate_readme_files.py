#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-10-01 11:13 下午

Author: huayang

Subject:

"""
import os
import re
import sys
import json
import inspect

from types import *
from typing import *
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path

os.environ['NUMEXPR_MAX_THREADS'] = '8'

WORK_UTILS = (10, 'Work Utils')
PYTORCH_MODELS = (20, 'Pytorch Models')
PYTORCH_UTILS = (30, 'Pytorch Utils')
PYTHON_UTILS = (40, 'Python Utils')

TAG_MAPPING = {
    'NLP Utils': WORK_UTILS,
    'Image Utils': WORK_UTILS,
    'Python Utils': PYTHON_UTILS,
    'Python 自定义数据结构': PYTHON_UTILS,
    'Pytorch Models': PYTORCH_MODELS,
    'Pytorch Utils': PYTORCH_UTILS,
    'Pytorch Loss': PYTORCH_UTILS,
    'Pytorch Train Plugin': PYTORCH_UTILS,
}


class args:  # noqa
    flag = 'huaytools'
    script_path = os.path.dirname(__file__)
    repo_path = os.path.abspath(os.path.join(script_path, '..'))
    src_path = os.path.join(repo_path, 'src')
    algo_path = os.path.join(repo_path, 'algorithms')
    prefix_topics = 'topics'
    prefix_problems = 'problems'
    prefix_notes = 'notes'
    problems_path = os.path.join(algo_path, prefix_problems)
    notes_path = os.path.join(algo_path, prefix_notes)
    topics_path = os.path.join(algo_path, prefix_topics)


sys.path.append(args.src_path)

try:
    from huaytools.python.code_analysis import module_iter, slugify
    from huaytools.python.file_utils import files_concat
    from huaytools.python.utils import get_logger
    from huaytools.tools.auto_readme import *
except:
    ImportError(f'import huaytools error.')

logger = get_logger()

RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.S)
RE_TAG = re.compile(r'Tag: (.*?)\s')
RE_SEP = re.compile(r'[,，、]')
RE_TITLE = re.compile(r'#+\s+(.*?)$')
RE_INDENT = re.compile(r'^([ ]*)(?=\S)', re.MULTILINE)

beg_details_tmp = '<details><summary><b> {key} <a href="{url}">¶</a></b></summary>\n'
beg_details_cnt_tmp = '<details><summary><b> {key} [{cnt}] <a href="{url}">¶</a></b></summary>\n'
end_details = '\n</details>\n'
auto_line = '<font color="LightGrey"><i> `This README is Auto-generated` </i></font>\n'


def build_tag2topic_map(notes_dir):
    file_names = os.listdir(notes_dir)

    topic2tags = dict()
    for fn in file_names:
        topic, _ = os.path.splitext(fn)
        txt = open(os.path.join(notes_dir, fn), encoding='utf8').read()
        tags = RE_SEP.split(RE_TAG.search(txt).group(1))
        topic2tags[topic] = tags
        # topic2tags[topic] = topic.split('-')[1].split('、')

    tag2topic = {v.lower(): k for k, vs in topic2tags.items() for v in vs}
    return tag2topic


# TODO: 对 `My Code Lab` 下的条目也添加上面的展示标签，和内部标签


def hn_line(line, lv=2):
    """"""
    return f'{"#" * lv} {line}'


class Algorithms:
    """"""
    sp_kw = {'题集', '模板', '经典'}
    more_info = '更多细分类型'

    def __init__(self):
        """"""
        self.args = args
        self.template_name = '*模板'
        self.toc_name = self.__class__.__name__
        self.prefix_topics = args.prefix_topics
        self.prefix_problems = args.prefix_problems
        self.prefix_notes = args.prefix_notes
        self.prefix_algorithm = os.path.basename(os.path.abspath(args.algo_path))
        # print(self.prefix_algo)
        self.prefix_algorithm_topics = os.path.join(self.prefix_algorithm, self.prefix_topics)
        self.prefix_algorithm_notes = os.path.join(self.prefix_algorithm, self.prefix_notes)
        # print(self.prefix_repo)

        # args.problems_path = os.path.join(args.algo_path, self.prefix_problems)
        # args.notes_path = os.path.join(args.algo_path, self.prefix_notes)
        self.tag2topic_map = build_tag2topic_map(args.notes_path)

        problems_dt = self.parse_problems()
        append_lines = self.gen_topic_md_sorted(problems_dt)
        self.content = '\n'.join(append_lines)

        diff = set(os.listdir(args.topics_path)) - set(os.listdir(args.notes_path))
        assert len(diff) == 0, diff
        # algo_path = os.path.join(repo_path, self.prefix)
        # fns = sorted([fn for fn in os.listdir(algo_path) if fn.startswith('专题-')])

        # toc_lns = [self.toc_head, '---']
        # for fn in fns:
        #     name, _ = os.path.splitext(fn)
        #     ln = f'- [{name}]({os.path.join(self.prefix, fn)})'
        #     toc_lns.append(ln)
        #
        # self.toc = '\n'.join(toc_lns)

    def gen_tags_svg(self, tags):  # noqa
        """"""
        lns = []
        for idx, (tag, topic) in enumerate(tags.items()):
            """"""
            # ![ForgiveDB](https://img.shields.io/badge/ForgiveDB-HuiZ-brightgreen.svg)
            lns.append(f'[![{tag}](https://img.shields.io/badge/{tag}-lightgray.svg)]({self.get_topic_fn(topic)})')
            # lns.append(f'[{tag}](https://img.shields.io/badge/{tag}-lightgray.svg)]')

        return '\n'.join(lns)

    def get_new_file_name(self, info):  # noqa
        """"""
        src, no, dif, name = info['来源'], info['编号'], info['难度'], info['标题']

        return f'{src}_{no}_{dif}_{name}.md'

    def parse_problems(self):
        """"""
        problems_dt = defaultdict(list)  # {tag: file_txt_ls}
        # files = os.listdir(args.problems_path)

        file_iter = []
        for prefix, _, files in os.walk(args.problems_path):
            for f in files:
                fn, ext = os.path.splitext(f)
                if ext != '.md' or fn.startswith('-'):
                    continue

                fp = os.path.join(prefix, f)
                suffix = '-'.join(prefix.split('/')[-2:])
                file_iter.append((fn, fp, suffix))

        # 解析算法 tags
        for fn, fp, suffix in file_iter:
            # fn, _ = os.path.splitext(f)
            # fp = os.path.join(args.problems_path, f)
            # src, pid, lv, pn = fn.rsplit('_', maxsplit=3)

            fp = Path(fp)
            txt = open(fp, encoding='utf8').read()
            info_ret = RE_INFO.search(txt)
            if not info_ret:
                print(fn, fp, suffix)
                continue

            info = json.loads(info_ret.group(1))

            # rename 如果需要
            new_file_name = self.get_new_file_name(info)
            if new_file_name != fp.name:
                logger.info(f'rename {fp.name} to {new_file_name}')
                fp = fp.rename(fp.parent / new_file_name)
                command_ln = f'git add "{fp}"'
                logger.info(command_ln)
                os.system(command_ln)

            src, pid, lv, pn = info['来源'], info['编号'], info['难度'], info['标题']
            tag_append = [src]  # if src != self.template_name else []

            # tags = RE_SEP.split(RE_TAG.search(txt).group(1)) + tag_append
            tags = info['tags'] + tag_append
            tags = [tag.strip() for tag in tags]
            tag2topic = {tag: self.tag2topic_map[tag.lower()] for tag in tags}
            topics = list(tag2topic.values())

            pid = f'No.{pid}' if pid.isnumeric() else pid
            head = f'`{src} {pid} {pn} ({lv}, {suffix})`'
            lines = txt.split('\n')
            # lines[0] = f'### {head}'
            lines.insert(0, '')
            lines.insert(0, self.gen_tags_svg(tag2topic))
            lines.insert(0, '')
            lines.insert(0, f'### {head}')
            txt = '\n'.join(lines)
            txt = txt.rstrip().replace(r'../../../_assets', '../_assets') + '\n\n---\n'
            for topic in topics:
                problems_dt[topic].append((head, txt))

        for k, v in problems_dt.items():
            problems_dt[k] = sorted(v)

        problems_dt = OrderedDict(sorted(problems_dt.items()))
        return problems_dt

    @staticmethod
    def get_topic_fn(tag):
        return f'{tag}.md'

    def gen_topic_md_sorted(self, problems_dt):
        """生成算法专题md，对主页topics排序"""
        readme_lines = [self.toc_name, '===\n', auto_line]
        append_lines = [self.toc_name, '---']

        append_blocks = []

        problems_index_ln = 'Problems Index'
        for tag, problems_txts in problems_dt.items():  # noqa
            """"""
            append_tmp = []
            topic_fn = self.get_topic_fn(tag)
            topic_name, _ = os.path.splitext(topic_fn)
            index_lines = [problems_index_ln, '---']
            # readme_lines.append(f'- [{topic_fn}]({topic_fn}.md)')
            # append_lines.append(f'- [{topic_fn}]({self.prefix}/{topic_fn}.md)')
            algo_url = os.path.join(self.prefix_topics, topic_fn)
            repo_url = os.path.join(self.prefix_algorithm_topics, topic_fn)
            problems_cnt = len(problems_txts)

            readme_lines.append(beg_details_cnt_tmp.format(key=topic_name, url=algo_url, cnt=problems_cnt))
            # append_lines.append(beg_details_tmp.format(key=topic_name, url=repo_url))
            append_tmp.append(beg_details_cnt_tmp.format(key=topic_name, url=repo_url, cnt=problems_cnt))

            contents = []
            for (head, txt) in problems_txts:
                # head = fn
                # link = self.parse_head(txt)
                link = slugify(head)
                contents.append(txt)
                index_lines.append(f'- [{head}](#{link})')
                readme_lines.append(f'- [{head}]({algo_url}#{link})')
                # append_lines.append(f'- [{head}]({repo_url}#{link})')
                append_tmp.append(f'- [{head}]({repo_url}#{link})')

            readme_lines.append(end_details)
            # append_lines.append(end_details)
            append_tmp.append(end_details)
            index_lines.append('\n---')

            topic_main_lines = open(os.path.join(args.repo_path, self.prefix_algorithm_notes, topic_fn),
                                    encoding='utf8').read().rstrip().split('\n')
            topic_main_lines.insert(0, f'[{problems_index_ln}](#{slugify(problems_index_ln)})\n')
            topic_main_lines.insert(0, f'# {tag.split("-")[1]}\n')

            topic_main = '\n'.join(topic_main_lines)
            topic_main_toc = '\n'.join(index_lines)
            topic_content = '\n'.join(contents)
            f_out = os.path.join(args.repo_path, self.prefix_algorithm_topics, topic_fn)
            content = files_concat([topic_main, topic_main_toc, topic_content], '\n')
            fw_helper.write(f_out, content)

            # topic_type = topic_name.split('-')[0]
            # append_blocks.append((append_tmp, topic_type, problems_cnt))
            append_blocks.append((append_tmp, topic_name, problems_cnt))

        # with open(os.path.join(args.algo_path, 'README.md'), 'w', encoding='utf8') as fw:
        #     fw.write('\n'.join(readme_lines))
        fw_helper.write(os.path.join(args.algo_path, 'README.md'), '\n'.join(readme_lines))

        # append_blocks = sorted(append_blocks, key=lambda x: (x[1], -x[2]))

        def block_assert(_block):
            return any(kw in _block[0] for kw in self.sp_kw)

        append_blocks = sorted(append_blocks)
        for it in append_blocks:
            block = it[0]
            if block_assert(block):
                append_lines += block

        append_lines.append('<details><summary><b>{more_info} ...<a href="{url}">¶</a></b></summary>\n'.format(
            more_info=self.more_info,
            url=f'{self.prefix_algorithm}/README.md'
        ))

        for it in append_blocks:
            block = it[0]
            if not block_assert(block):
                append_lines += block

        append_lines.append(end_details)

        # append_lines.append(f'- [All Topics]({self.prefix_algo}/README.md)')
        return append_lines

    @staticmethod
    def parse_head(txt):
        """"""
        # 标题解析
        try:
            head = RE_TITLE.search(txt.split('\n', maxsplit=1)[0]).group(1)
        except:
            raise Exception('parsing head error!')

        return head


class Codes:
    """"""

    @dataclass()
    class DocItem:
        """ 每个 docstring 需要提取的内容 """
        flag: Tuple
        summary: str
        content: str
        module_path: str
        line_no: int
        link: str = None

        def __post_init__(self):
            self.link = f'[source]({self.module_path}#L{self.line_no})'

        def get_block(self, prefix=''):
            """"""

            block = f'### {self.summary}\n'
            block += f'> [source]({os.path.join(prefix, self.module_path)}#L{self.line_no})\n\n'
            # block += f'<details><summary><b> Intro & Example </b></summary>\n\n'
            block += '```python\n'
            block += f'{self.content}'
            block += '```\n'
            # block += '\n</details>\n'

            return block

    def __init__(self):
        """"""
        # self.code_path = args.code_path
        # print(self.code_path)
        self.code_readme_path = os.path.join(args.src_path, 'README.md')
        self.toc_name = self.__class__.__name__
        docs_dt = self.parse_docs()
        self.code_basename = os.path.basename(os.path.abspath(args.src_path))
        self.content = self.gen_readme_md_simply(docs_dt)

    def parse_docs(self):
        """ 生成 readme for code """
        docs_dt = defaultdict(list)

        sys.path.append(args.repo_path)
        for module in module_iter(args.src_path):
            if hasattr(module, '__all__'):
                # print(module.__name__)
                for obj_str in module.__all__:
                    obj = getattr(module, obj_str)
                    if isinstance(obj, (ModuleType, FunctionType, type)) \
                            and getattr(obj, '__doc__') \
                            and obj.__doc__.startswith('@'):
                        # print(obj.__name__)
                        doc = self.parse_doc(obj)
                        docs_dt[doc.flag].append(doc)

        return docs_dt

    def parse_doc(self, obj) -> DocItem:
        """"""
        raw_doc = obj.__doc__
        lines = raw_doc.split('\n')
        flag = TAG_MAPPING[lines[0][1:]]

        lines = lines[1:]
        min_indent = self.get_min_indent('\n'.join(lines))
        lines = [ln[min_indent:] for ln in lines]

        summary = f'`{obj.__name__}: {lines[0]}`'
        content = '\n'.join(lines)

        line_no = self.get_line_number(obj)
        module_path = self.get_module_path(obj)
        return self.DocItem(flag, summary, content, module_path, line_no)

    @staticmethod
    def get_line_number(obj):
        """ 获取对象行号
        基于正则表达式，所以不一定保证准确
        """
        return inspect.findsource(obj)[1] + 1

    @staticmethod
    def get_module_path(obj):
        abs_url = inspect.getmodule(obj).__file__
        dirs = abs_url.split('/')
        idx = dirs[::-1].index(args.flag)  # *从后往前*找到 my 文件夹，只有这个位置是基本固定的
        return '/'.join(dirs[-(idx + 1):])  # 再找到这个 my 文件夹的上一级目录

    @staticmethod
    def get_min_indent(s):
        """Return the minimum indentation of any non-blank line in `s`"""
        indents = [len(indent) for indent in RE_INDENT.findall(s)]
        if len(indents) > 0:
            return min(indents)
        else:
            return 0

    def gen_readme_md_simply(self, docs_dt: Dict[str, List[DocItem]]):
        """ 简化首页的输出 """
        # args = self.args
        # code_prefix = os.path.basename(os.path.abspath(args.code_path))
        # print(code_prefix)

        toc = [self.toc_name, '---']
        append_toc = [self.toc_name, '---']
        readme_lines = []
        # append_lines = []

        key_sorted = sorted(docs_dt.keys())
        for key in key_sorted:
            blocks = docs_dt[key]
            key = key[1]
            toc.append(beg_details_tmp.format(key=key, url=f'#{slugify(key)}'))

            # append_toc.append(beg_details_tmp.format(key=key, url=f'{self.code_basename}/README.md#{slugify(key)}'))
            append_toc.append('### {key} [¶]({url})\n'.format(key=key,
                                                              url=f'{self.code_basename}/README.md#{slugify(key)}'))

            readme_lines.append(hn_line(key, 2))
            # append_lines.append(hn_line(key, 2))
            for d in blocks:
                toc.append(f'- [{d.summary}](#{slugify(d.summary)})')
                append_toc.append(f'- [{d.summary}]({self.code_basename}/README.md#{slugify(d.summary)})')
                readme_lines.append(d.get_block())
                # append_lines.append(d.get_block(prefix=code_prefix))

            toc.append(end_details)

            # append_toc.append(end_details)
            append_toc.append('\n')

        toc_str = '\n'.join(toc[:2] + [auto_line] + toc[2:])
        sep = '\n---\n\n'
        content_str = '\n\n'.join(readme_lines)
        code_readme = toc_str + sep + content_str
        # with open(self.code_readme_path, 'w', encoding='utf8') as fw:
        #     fw.write(code_readme)
        fw_helper.write(self.code_readme_path, code_readme)

        append_toc_str = '\n'.join(append_toc)
        main_append = append_toc_str + sep  # + '\n\n'.join(append_lines)
        return main_append


def get_repo_toc(*toc_parts):
    """"""
    lns = ['Repo Index', '---']
    for part in toc_parts:
        name = part.toc_name
        lns.append(f'- [{name}](#{slugify(name)})')
    return '\n'.join(lns)


# TOTAL_ADD = 0


# def file_write_helper(abspath, content):
#     """"""
#     global TOTAL_ADD
#
#     old_content = ''
#     if os.path.exists(abspath):
#         old_content = open(abspath, encoding='utf8').read()
#
#     if old_content != content:
#         with open(abspath, 'w', encoding='utf8') as fw:
#             fw.write(content)
#
#         command_ln = f'git add "{abspath}"'
#         logger.info(command_ln)
#         os.system(command_ln)
#         TOTAL_ADD += 1


def pipeline():
    """"""
    # args = simple_argparse()
    args.repo_readme_path = os.path.join(args.repo_path, r'README.md')
    # if os.path.exists(args.repo_readme_path):
    #     readme_old = open(args.repo_readme_path, encoding='utf8').read()
    # else:
    #     readme_old = ''
    # code_toc, code_append = gen_code_readme(args)

    parts = [
        Algorithms(),
        Notes(),
        Papers(),
        Books(),
        Codes()
    ]
    repo_toc = get_repo_toc(*parts)
    readme_main_path = os.path.join(args.repo_path, r'README-main.md')
    main_auto_line = '<font color="LightGrey"><i> `The following is Auto-generated` </i></font>'
    content = files_concat(src_in=[readme_main_path,
                                   # main_auto_line,
                                   repo_toc] + [it.content for it in parts],
                           sep='\n---\n\n')
    fw_helper.write(args.repo_readme_path, content)
    # readme = open(args.repo_readme_path, encoding='utf8').read()
    # if readme_old != readme:
    print(fw_helper.add_cnt)


if __name__ == '__main__':
    """"""
    pipeline()

    # if len(sys.argv) > 1:
    #     pipeline()
    #     # print('SUCCESS')
    # else:
    #     # 抑制标准输出，只打印 WARNING 信息
    #     # sys.stdout = open(os.devnull, 'w')
    #     command = "generate_readme_files.py " \
    #               "--repo_path ../ " \
    #               "--code_path ../code/ " \
    #               "--algo_path ../algorithm/ "
    #     sys.argv = command.split()
    #     _test()
