#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-06-29 3:35 下午
    
Author: huayang
    
Subject: Utils for Pytorch

"""
import os
import doctest
import random

from typing import *
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

import numpy as np

from torch import Tensor

from huaytools.python.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'set_seed',
    'apply_to',
    'cosine_similarity_dense',
    'init_weights',
    'get_torch_version',
    'load_state_dict_tf',
    'load_state_dict_pt',
    'load_state_dict_explicit',
    'default_device',
    'set_device',
    'set_device_cpu'
]


def set_seed(seed: int = None, apply_cudnn=True):
    """@Pytorch Utils
    设置全局随机数种子，使实验可复现

    Args:
        seed:
        apply_cudnn: cudnn 对卷积操作进行了优化，牺牲了精度来换取计算效率；如果对精度要求不高，可以设置为 False

    Notes:
        （似乎不是必要的）如果在 DataLoader 设置了 num_workers>0，还需要设置 worker_init_fn，以确保数据加载的顺序；
            ```
            def _worker_init_fn(worker_id):
                np.random.seed(int(seed) + worker_id)
            ```

    References:
        [PyTorch固定随机数种子](https://blog.csdn.net/john_bh/article/details/107731443)
    """
    if seed is None:
        return

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # noqa, 为了禁止hash随机化，使得实验可复现

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

    if apply_cudnn:
        torch.backends.cudnn.benchmark = False  # noqa
        torch.backends.cudnn.deterministic = True  # noqa


def apply_to(obj, sth):
    """

    Examples:
        >>> t = torch.as_tensor([1,2,3])
        >>> t.dtype
        torch.int64
        >>> t = apply_to(t, float)
        >>> t.dtype
        torch.float64
        >>> ts = [torch.as_tensor([1,2,3]), torch.as_tensor([4,5,6])]
        >>> ts = apply_to(ts, float)
        >>> [t.dtype for t in ts]
        [torch.float64, torch.float64]
        >>> ts = {'a': torch.as_tensor([1]), 'b': [torch.as_tensor([2]), {'c': torch.as_tensor([3])}]}
        >>> ts = apply_to(ts, float)
        >>> [ts['a'].dtype, ts['b'][0].dtype, ts['b'][1]['c'].dtype]
        [torch.float64, torch.float64, torch.float64]

    """
    if hasattr(obj, "to"):
        return obj.to(sth)
    elif isinstance(obj, (List, Tuple)):
        return type(obj)(apply_to(o, sth) for o in obj)
    elif isinstance(obj, Mapping):
        new_obj = [(k, apply_to(v, sth)) for k, v in obj.items()]
        return type(obj)(new_obj)  # noqa
    else:
        raise TypeError(
            f"Can't apply {apply_to.__name__} on object of type {type(obj)}, "
            f"only of nested list/tuple/dicts of objects "
        )


def cosine_similarity_dense(x1, x2):
    """ cosine 距离（全连接）
        即 x1 中每个向量与 x2 中每个向量计算 cosine 距离，相当于计算一个 attention 矩阵

        等价于 `F.cosine_similarity(x1.unsqueeze(1), x1.unsqueeze(0), dim=-1)`
    Args:
        x1: [B1, N]
        x2: [B2, N]

    Returns:
        [B1, B2] matrix
    """
    from ..backend import l2_normalize

    assert x1.ndim == x2.ndim == 2

    x1_normalized = l2_normalize(x1, dim=-1)  # [B1, N]
    x2_normalized_T = l2_normalize(x2, dim=-1).T  # [N, B2]
    return torch.matmul(x1_normalized, x2_normalized_T)  # [B1, B2]


def create_mask_3d(q_tensor: Tensor, v_mask: Tensor, dtype=torch.float):
    """ Create 3D attention mask from a 2D tensor mask.

    Args:
      q_tensor: 2D or 3D Tensor of shape [B, Q, ...].
      v_mask: int32 Tensor of shape [B, V].
      dtype:

    Returns:
        float Tensor of shape [B, Q, V].

    References:
        [google-research/bert](https://github.com/google-research/bert)
    """
    B = q_tensor.shape[0]  # B
    Q = q_tensor.shape[1]  # Q

    v_mask = v_mask.unsqueeze(1)  # [B, V] -> [B, 1, V]
    mask = torch.ones([B, Q, 1]) * v_mask  # [B, Q, V]
    return mask.to(dtype)


def init_weights(module: nn.Module, normal_std=0.02):
    """@Pytorch Utils
    默认参数初始化

    Examples:
        >>> model = nn.Transformer()
        >>> _ = model.apply(init_weights)

    Args:
        module:
        normal_std:

    References: Bert
    """
    if isinstance(module, nn.Linear):
        # truncated_normal
        nn.init.trunc_normal_(module.weight.data, std=normal_std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        # truncated_normal
        nn.init.trunc_normal_(module.weight.data, std=normal_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
    else:
        pass  # default


def get_torch_version():
    """"""
    return torch.__version__


def load_state_dict_tf(weights_path):
    """"""
    import tensorflow as tf

    def _loader(name):
        """"""
        return tf.train.load_variable(weights_path, name)

    if os.path.isdir(weights_path):  # 如果是目录
        # 找出目录下的 xxx.ckpt.index 文件
        file_ls = os.listdir(weights_path)
        file_name = [f for f in file_ls if f.endswith('.index')][0]
        weights_path = os.path.join(weights_path, file_name)

    weights_path = weights_path[:-6] if weights_path.endswith('.index') else weights_path
    weights_pretrained = OrderedDict()
    for n, _ in tf.train.list_variables(weights_path):
        array = _loader(n)
        if n.endswith('kernel'):
            array = np.transpose(array)  # transpose(tf[in, out]) -> pt[out, in]
        weights_pretrained[n] = torch.as_tensor(array)

    return weights_pretrained


def load_state_dict_pt(weights_path, map_location='cpu'):
    """"""
    state_dict = torch.load(weights_path, map_location=map_location)
    return state_dict


def load_state_dict_explicit(model: nn.Module, state_dict, name_mapping=None):
    """ 与 m.load_state_dict 功能类似，对未加载的权重给出更明确的提示

    Args:
        model:
        state_dict: {name: tensor} 字典
        name_mapping: {name: name_old} 字典，默认为 None；
            当 weights_dict 与模型中的权重名称不匹配时，可以通过 name_mapping 再映射一次

    Examples:
        >>> m = nn.Linear(3, 4)  # {'weight': ..., 'bias': ...}
        >>> wd = {'w': torch.randn(4, 3), 'b': torch.randn(4)}
        >>> nm = {'weight': 'w', 'bias': 'b'}
        >>> _ = load_state_dict_explicit(m, wd, nm)
    """
    if name_mapping:
        for name, name_old in name_mapping.items():
            if name_old in state_dict:
                state_dict[name] = state_dict.pop(name_old)  # 替换新名称

    load_keys = set()  # 记录顺利加载的 key
    state_dict_tmp = OrderedDict()  # 新 state_dict，不直接修改原 state_dict
    state_dict_old = model.state_dict()
    for name, tensor in state_dict_old.items():
        if name not in state_dict:
            state_dict_tmp[name] = tensor
        else:
            _assert_shape(state_dict[name], tensor)  # noqa

            state_dict_tmp[name] = state_dict[name]
            load_keys.add(name)

    missed_keys = sorted(set(state_dict_old.keys()) - load_keys)  # 未更新的权重
    unused_keys = sorted(set(state_dict.keys()) - load_keys)  # 未使用的权重
    logger.info(f'Missed weights({len(missed_keys)}): {missed_keys}')
    logger.info(f'Unused weights({len(unused_keys)}): {unused_keys}')

    model.load_state_dict(state_dict_tmp)  # reload
    model.eval()  # deactivate dropout
    return model


def log_softmax(x: Tensor, dim=-1):
    """"""
    x = softmax(x, dim=dim)  # [B, C]
    return torch.log(x)  # [B, C]


def sequence_masking(x: torch.Tensor,
                     mask: torch.Tensor,
                     axis=1, mode='add', inf=1e12):
    """序列 mask

    Args:
        x: 2D 或 2D 以上张量，必须包含 batch_size 和 seq_len 两个维度
        mask: 形如  (batch_size, seq_len) 的 0/1 矩阵
        axis: 需要 mask 的维度，即 seq_len 所在维度，默认为 1
        mode: 有 'mul' 和 'add' 两种：
            mul 会将 pad 部分置零，一般用于全连接层之前；
            add 会把 pad 部分减去一个大的常数，一般用于 softmax 之前。
        inf: 大的常数

    Returns:
        tensor with shape same as x

    Examples:
        mask = [B, L]
        示例 1：x.shape = [B, L, _],     则 axis=1 (默认)
        示例 2：x.shape = [B, _, L, _],  则 axis=2
        示例 3：x.shape = [B, _, _, L],  则 axis=-1
    """
    if mask is None:
        return x

    assert mask.ndim == 2, 'only for mask.ndim == 2'

    if axis < 0:
        axis = x.ndim + axis

    # 将 mask 的维度扩充到与 x 一致，以便进行广播
    # 示例：假设 x.shape = [B, _, L, _]
    # 则经过以下操作后，mask.shape = [B, 1, L, 1]，相当于 mask = mask[:, None, :, None]
    for _ in range(axis - 1):
        mask = mask.unsqueeze(1)
    for _ in range(x.ndim - axis - 1):
        mask = mask.unsqueeze(-1)

    if mode == 'mul':
        return x * mask
    elif mode == 'add':
        return x - (1 - mask) * inf
    else:
        raise ValueError('`mode` must be one of %s' % {'add', 'mul'})


def softmax(x: Tensor, dim=-1):
    """"""
    x_exp = torch.exp(x)  # [B, C]
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)  # [B, 1]
    return x_exp / x_exp_sum  # [B, C]


def _assert_shape(tensor1, tensor2):
    """"""
    t1_shape = list(tensor1.shape)
    t2_shape = list(tensor2.shape)
    assert t1_shape == t2_shape, f'shape mismatching: {t1_shape} vs {t2_shape}'
    return True


_DEFAULT_DEVICE_STR: str = '_DEFAULT_DEVICE'


def default_device() -> str:
    """
    Examples:
        >>> default_device()
        'cpu'
        >>> set_device('xxx')
        >>> default_device()
        'xxx'
        >>> set_device_cpu()
        >>> default_device()
        'cpu'
    """
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return os.environ.get(_DEFAULT_DEVICE_STR, 'cuda' if torch.cuda.is_available() else 'cpu')


def set_device_cpu():
    """
    Examples:
        >>> set_device_cpu()
        >>> default_device()
        'cpu'

    """
    set_device('cpu')


def set_device(device: str):
    """
    Examples:
        >>> set_device('aaa')
        >>> default_device()
        'aaa'

    """
    # global _DEFAULT_DEVICE
    # _DEFAULT_DEVICE = device
    os.environ[_DEFAULT_DEVICE_STR] = device


def _test():
    """"""
    doctest.testmod()


if __name__ == '__main__':
    """"""
    _test()
