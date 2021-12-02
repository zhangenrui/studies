#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2021-11-19 11:07 上午

Author: huayang

Subject:

"""
import os  # noqa
import doctest  # noqa

# from itertools import islice
# from collections import defaultdict

# from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

__all__ = [
    'DictTensorDataset',
    'ToyDataLoader'
]


class DictTensorDataset(Dataset[Dict[str, Tensor]]):  # python=3.6中使用，需删掉 [Dict[str, Tensor]]
    """@Pytorch Utils
    字典格式的 Dataset

    Examples:
        >>> x = y = torch.as_tensor([1,2,3,4,5])
        >>> _ds = DictTensorDataset(x=x, y=y)
        >>> len(_ds)
        5
        >>> dl = DataLoader(_ds, batch_size=3)
        >>> for batch in dl: print(batch)
        {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
        {'x': tensor([4, 5]), 'y': tensor([4, 5])}
        >>> len(dl)
        2

    References:
        - torch.utils.data.TensorDataset
        - huggingface/datasets.arrow_dataset.Dataset
    """

    tensors_dict: Dict[str, Tensor]

    def __init__(self, **tensors_dict: Tensor) -> None:
        """
        Args:
            **tensors_dict:
        """
        assert len(np.unique([tensor.shape[0] for tensor in tensors_dict.values()])) == 1, \
            "Size mismatch between tensors"
        self.tensors_dict = tensors_dict

    def __getitem__(self, index) -> Dict[str, Tensor]:
        """"""
        return {name: tensor[index] for name, tensor in self.tensors_dict.items()}

    def __len__(self):
        return next(iter(self.tensors_dict.values())).shape[0]


def _test():
    """"""
    doctest.testmod()

    d = []
    for i in range(10):
        d.append([[i, i], [i * i] * 2, [i ** 3] * 3])

    ds = ListDataset(d)
    print(len(ds))

    dl = DataLoader(ds, 2, num_workers=2)
    for b in dl:
        print(b)


if __name__ == '__main__':
    """"""
    _test()


class ToyDataLoader(DataLoader):
    """@Pytorch Utils
    简化创建 DataLoader 的过程

    Examples:
        # single input
        >>> x = [1,2,3,4,5]
        >>> dl = ToyDataLoader(x, batch_size=3, single_input=True, shuffle=False)
        >>> for batch in dl:
        ...     print(type(batch).__name__, batch)
        list [tensor([1, 2, 3])]
        list [tensor([4, 5])]

        # multi inputs
        >>> x = y = [1,2,3,4,5]
        >>> dl = ToyDataLoader([x, y], batch_size=3, shuffle=False, device='cpu')
        >>> for batch in dl:
        ...     print(type(batch).__name__, batch)
        list [tensor([1, 2, 3]), tensor([1, 2, 3])]
        list [tensor([4, 5]), tensor([4, 5])]

        # multi inputs (dict)
        >>> x = y = [1,2,3,4,5]
        >>> dl = ToyDataLoader({'x': x, 'y': y}, batch_size=3, shuffle=False, device='cpu')
        >>> for batch in dl:
        ...     print(type(batch).__name__, batch)
        dict {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
        dict {'x': tensor([4, 5]), 'y': tensor([4, 5])}

        # multi inputs (row2col)
        >>> xy = [[1,1],[2,2],[3,3],[4,4],[5,5]]
        >>> dl = ToyDataLoader(xy, batch_size=3, row2col=True, shuffle=False, device='cpu')
        >>> for batch in dl:
        ...     print(type(batch).__name__, batch)
        list [tensor([1, 2, 3]), tensor([1, 2, 3])]
        list [tensor([4, 5]), tensor([4, 5])]

        # multi inputs (dict, row2col)
        >>> xy = [{'x':1,'y':1},{'x':2,'y':2},{'x':3,'y':3},{'x':4,'y':4},{'x':5,'y':5}]
        >>> dl = ToyDataLoader(xy, batch_size=3, row2col=True, shuffle=False, device='cpu')
        >>> for batch in dl:
        ...     print(type(batch).__name__, batch)
        dict {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
        dict {'x': tensor([4, 5]), 'y': tensor([4, 5])}

    Notes:
        V1: 当数据较大时，直接把所有数据 to('cuda') 会爆内存，所以删除了 default_device
            如果数据量比较小，也可以设置 device='cuda' 提前把数据移动到 GPU
        V2: 重写了 __iter__()，在产生 batch 时才移动 tensor，因此还原了 default_device
    """

    def __init__(self, dataset: Iterable, batch_size,
                 single_input=False, row2col=False,
                 shuffle=True, device=None, **kwargs):
        """

        Args:
            dataset:
            batch_size:
            single_input: if is single input, default False
            row2col: work when `single_input` is False, default False
            shuffle: default True
            device: default None
            **kwargs:
        """
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if single_input:
            dataset = TensorDataset(torch.as_tensor(dataset))
        else:  # multi inputs
            if row2col:
                if isinstance(next(iter(dataset)), Dict):
                    col_dataset = defaultdict(list)
                    for row in dataset:
                        for k, v in row.items():
                            col_dataset[k].append(v)
                    dataset = col_dataset
                else:
                    dataset = zip(*dataset)

            if isinstance(dataset, Dict):
                dataset = {name: torch.as_tensor(tensor) for name, tensor in dataset.items()}
                dataset = DictTensorDataset(**dataset)
            else:
                dataset = [torch.as_tensor(tensor) for tensor in list(dataset)]
                dataset = TensorDataset(*dataset)

        # 这样写会导致无法实用其他类型的 sampler
        # if shuffle:
        #     sampler = RandomSampler(dataset, replacement=repeat_sampling)
        # else:
        #     sampler = SequentialSampler(dataset)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __iter__(self):
        """
        References:
            from hanlp.common.dataset import DeviceDataLoader
        """
        for batch in super().__iter__():
            if self.device is not None:
                if isinstance(batch, Dict):
                    batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                else:
                    batch = [tensor.to(self.device) for tensor in batch]

            yield batch