#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-08-31 9:33 下午

Author:
    huayang

Subject:

"""
import os
import sys
import json
import time
import doctest
import logging
import platform
import functools

import requests

from datetime import datetime
from typing import Any

DEFAULT_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=DEFAULT_LOG_FMT, datefmt='%Y.%m.%d %H:%M:%S', level=logging.INFO)

__all__ = [
    'get_logger',
    'get_caller_name',
    'set_stdout_null',
    'set_env',
    'get_env',
    'set_attr',
    'get_attr',
    'set_default',
    'get_print_json',
    'get_response',
    'get_time_string',
    'get_system_type',
    'enable_function_test',
    'function_test_dn',
    'get_typename',
    'is_mac',
    'is_linux',
    'is_windows'
]


def get_caller_name(num_back=2) -> str:
    """@Python Utils
    获取调用者的名称

    如果是方法，则返回方法名；
    如果是模块，则返回文件名；
    如果是类，返回类名，但要作为类属性，而不是定义在 __init__ 中

    说明：如果在方法内使用，那么直接调用 `sys._getframe().f_code.co_name` 就是输出了本身的函数名；
        这里因为是作为工具函数，所以实际上输出的调用本方法的函数名，所以需要 `f_back` 一次

    Args:
        num_back: 回溯层级，大于 0，默认为 2

    Examples:
        >>> def f():  # 不使用本方法
        ...     return sys._getframe().f_code.co_name  # noqa
        >>> f()
        'f'
        >>> def foo():
        ...     return get_caller_name(1)
        >>> foo()
        'foo'

        # 使用场景：查看是谁调用了 `bar` 方法
        >>> def bar():
        ...     return get_caller_name()
        >>> def zoo():
        ...     return bar()
        >>> zoo()
        'zoo'

        # 使用场景：自动设置 logger name
        >>> def _get_logger(name=None):
        ...     name = name or get_caller_name()
        ...     return logging.getLogger(name)
        >>> class T:
        ...     cls_name = get_caller_name(1)  # level=1
        ...     logger = _get_logger()  # get_logger 中使用了 get_caller_name
        >>> T.cls_name
        'T'
        >>> T.logger.name
        'T'

        # 使用场景：自动从字典中获取属性值
        >>> class T:
        ...     default = {'a': 1, 'b': 2}
        ...     def _get_attr(self):
        ...         name = get_caller_name()
        ...         return self.default[name]
        ...     @property
        ...     def a(self):
        ...         # return default['a']
        ...         return self._get_attr()
        ...     @property
        ...     def b(self):
        ...         # return default['b']
        ...         return self._get_attr()
        >>> t = T()
        >>> t.a
        1
        >>> t.b
        2

    """
    assert num_back >= 1

    frame = sys._getframe()  # noqa
    while num_back > 0:
        frame = frame.f_back
        num_back -= 1

    co_name = frame.f_code.co_name

    if co_name == '<module>':  # 当调用方是一个模块，此时返回模块的文件名
        filename, _ = os.path.splitext(os.path.basename(frame.f_code.co_filename))
        return filename

    return co_name


def set_stdout_null():
    """ 抑制标准输出 """
    sys.stdout = open(os.devnull, 'w')


def get_print_json(obj, **json_kwargs):
    """ 生成 printable json"""
    from huaytools.python.custom import AnyEncoder

    obj = obj if isinstance(obj, dict) else obj.__dict__

    json_kwargs.setdefault('cls', AnyEncoder)
    json_kwargs.setdefault('indent', 4)
    json_kwargs.setdefault('ensure_ascii', False)
    json_kwargs.setdefault('sort_keys', True)
    return json.dumps(obj, **json_kwargs)


def set_default(obj, name: str, default: Any) -> Any:
    """ 行为类似 dict.setdefault，可以作用于一般类型（兼容 dict） """
    if isinstance(obj, dict):
        return obj.setdefault(name, default)

    if not hasattr(obj, name):
        return setattr(obj, name, default)
    return getattr(obj, name)


def get_attr(args, name: str, default=None) -> Any:
    """ 等价于 getattr（兼容 dict）；跟 set_default 的区别是，如果 obj 中不存在 name 这个参数，不会将其添加到对象中 """
    if isinstance(args, dict):
        if default is not None:
            return args.get(name, default)
        else:
            return args[name]  # args.get 不会报异常
    else:
        if default is not None:
            return getattr(args, name, default)
        else:
            return getattr(args, name)


def set_attr(args, name: str, value) -> None:
    """ 等价于 setattr（兼容 dict） """
    if isinstance(args, dict):
        args[name] = value
    else:
        setattr(args, name, value)


def get_typename(o):
    """
    References: torch.typename
    """
    module = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name


def set_env(key: str, value: str):
    """ 设置环境变量 """
    os.environ[key] = value


def get_env(key, default=None):
    """
    Examples:
        >>> get_env('HOME')
        '/Users/huayang'
    """
    return os.environ.get(key, default)


def get_env_dict():
    """ 获取环境变量（字典）

    Examples:
        >>> env = get_env_dict()
        >>> env['ttt'] = 'ttt'
        >>> env['ttt']
        'ttt'
    """
    return os.environ


def get_logger(name=None):
    """"""
    name = name or get_caller_name()
    return logging.getLogger(name)


def get_time_string(fmt="%Y%m%d%H%M%S"):
    """获取当前时间（格式化）"""
    return datetime.now().strftime(fmt)


def get_system_type():
    """获取当前系统类型"""
    return platform.system()


def _system_is(sys_name: str):
    """"""
    sys_name = sys_name.lower()
    if sys_name in {'mac', 'macos'}:
        sys_name = 'Darwin'
    elif sys_name in {'win', 'window', 'windows'}:
        sys_name = 'Windows'

    return get_system_type().lower() == sys_name


def is_mac():
    """判断是否为 mac os 系统"""
    return _system_is('Darwin')


def is_linux():
    """判断是否为 linux 系统"""
    return _system_is('Linux')


def is_windows():
    """判断是否为 windows 系统"""
    return _system_is('Windows')


def get_response(url,
                 timeout=3,
                 n_retry_max=5,
                 return_content=True,
                 check_func=None):
    """
    Args:
        url:
        timeout: 超时时间，单位秒
        n_retry_max: 最大重试次数
        return_content: 是否返回 response.content
        check_func: 内容检查函数，函数接收单个 response 对象作为参数
    """
    n_retry = 0
    response = None
    while n_retry < n_retry_max:
        try:
            response = requests.get(url=url, timeout=timeout)
            if return_content:
                response = response.content
            if check_func is None or check_func(response):
                break
        except:
            pass
        finally:
            n_retry += 1

    return response


def download_file(url,
                  save_path=None,
                  **kwargs):
    """
    下载指定 url 内容

    Args:
        url:
        save_path: 保存路径
        kwargs: get_response 相关参数

    """
    kwargs['return_content'] = kwargs.pop('return_content', False)
    response = get_response(url, **kwargs)

    if response and save_path:
        with open(save_path, mode='wb') as f:
            f.write(response.content)

    return save_path


FN_TEST_ENV = '_function_test'


def enable_function_test():
    """"""
    set_env(FN_TEST_ENV, '1')


def function_test_dn(func):
    """@Python Utils
    函数测试装饰器

    Examples:
        >>> enable_function_test()
        >>> @function_test_dn
        ... def _test_func(x=1):
        ...     print(x)
        >>> _test_func()
        Start running `_test_func` {
        1
        } End, spend 0 s.
        <BLANKLINE>
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        """"""
        if get_env(FN_TEST_ENV, '0') != '1':
            return

        print('Start running `%s` {' % func.__name__)

        start = time.time()

        func(*args, **kwargs)

        end = time.time()

        print('} End, spend %s s.\n' % round(end - start))

    return inner


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
