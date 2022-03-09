Codes
---
<font color="LightGrey"><i> `This README is Auto-generated` </i></font>

<details><summary><b> Work Utils <a href="#work-utils">¶</a></b></summary>

- [`find_best_threshold: 搜索最佳阈值（二分类）`](#find_best_threshold-搜索最佳阈值二分类)
- [`BertTokenizer: Bert 分词器`](#berttokenizer-bert-分词器)
- [`ner_result_parse: NER 结果解析（基于 BIO 格式）`](#ner_result_parse-ner-结果解析基于-bio-格式)
- [`split: 将数据按比例切分`](#split-将数据按比例切分)
- [`XLSHelper: Excel 文件加载（基于 openpyxl）`](#xlshelper-excel-文件加载基于-openpyxl)
- [`ImageCheck: 图片完整性检查`](#imagecheck-图片完整性检查)
- [`get_real_ext: 获取图像文件的真实后缀`](#get_real_ext-获取图像文件的真实后缀)

</details>

<details><summary><b> Pytorch Models <a href="#pytorch-models">¶</a></b></summary>

- [`DualNet: 双塔结构`](#dualnet-双塔结构)
- [`SiameseNet: 孪生网络，基于双塔结构`](#siamesenet-孪生网络基于双塔结构)
- [`SimCSE: SimCSE`](#simcse-simcse)
- [`Bert: Bert by Pytorch`](#bert-bert-by-pytorch)
- [`BertPretrain: Bert 预训练（MLM + NSP）`](#bertpretrain-bert-预训练mlm-nsp)
- [`SentenceBert: Bert 句向量`](#sentencebert-bert-句向量)
- [`BertSequenceTagging: Bert 序列标注`](#bertsequencetagging-bert-序列标注)
- [`BertTextClassification: Bert 文本分类`](#berttextclassification-bert-文本分类)
- [`LayerNorm: Layer Normalization`](#layernorm-layer-normalization)

</details>

<details><summary><b> Pytorch Utils <a href="#pytorch-utils">¶</a></b></summary>

- [`DictTensorDataset: 字典格式的 Dataset`](#dicttensordataset-字典格式的-dataset)
- [`ToyDataLoader: 简化创建 DataLoader 的过程`](#toydataloader-简化创建-dataloader-的过程)
- [`BertDataLoader: 简化 Bert 训练数据的加载`](#bertdataloader-简化-bert-训练数据的加载)
- [`ContrastiveLoss: 对比损失（默认距离函数为欧几里得距离）`](#contrastiveloss-对比损失默认距离函数为欧几里得距离)
- [`CrossEntropyLoss: 交叉熵`](#crossentropyloss-交叉熵)
- [`TripletLoss: Triplet 损失，常用于无监督学习、few-shot 学习`](#tripletloss-triplet-损失常用于无监督学习few-shot-学习)
- [`FGM: Fast Gradient Method (对抗训练)`](#fgm-fast-gradient-method-对抗训练)
- [`PGM: Projected Gradient Method (对抗训练)`](#pgm-projected-gradient-method-对抗训练)
- [`Trainer: Trainer 基类`](#trainer-trainer-基类)
- [`set_seed: 设置全局随机数种子，使实验可复现`](#set_seed-设置全局随机数种子使实验可复现)
- [`init_weights: 默认参数初始化`](#init_weights-默认参数初始化)
- [`mixup: mixup 数据增强策略`](#mixup-mixup-数据增强策略)

</details>

<details><summary><b> Python Utils <a href="#python-utils">¶</a></b></summary>

- [`simple_argparse: 一个简化版 argparse`](#simple_argparse-一个简化版-argparse)
- [`ArrayDict: 数组字典，支持 slice`](#arraydict-数组字典支持-slice)
- [`ValueArrayDict: 数组字典，支持 slice，且操作 values`](#valuearraydict-数组字典支持-slice且操作-values)
- [`BunchDict: 基于 dict 实现 Bunch 模式`](#bunchdict-基于-dict-实现-bunch-模式)
- [`FieldBunchDict: 基于 dataclass 的 BunchDict`](#fieldbunchdict-基于-dataclass-的-bunchdict)
- [`ls_dir_recur: 递归遍历目录下的所有文件`](#ls_dir_recur-递归遍历目录下的所有文件)
- [`files_concat: 文件拼接`](#files_concat-文件拼接)
- [`get_caller_name: 获取调用者的名称`](#get_caller_name-获取调用者的名称)
- [`function_test_dn: 函数测试装饰器`](#function_test_dn-函数测试装饰器)

</details>

---

## Work Utils

### `find_best_threshold: 搜索最佳阈值（二分类）`
> [source](huaytools/metrics/utils.py#L40)

```python
搜索最佳阈值（二分类）

Examples:
    >>> _scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    >>> _labels = [0, 0, 1, 0, 1, 1, 1, 1]
    >>> o = find_best_threshold(_scores, _labels)
    >>> o.best_accuracy, o.best_accuracy_threshold
    (0.875, 0.45)
    >>> o.best_f1, o.best_f1_threshold, o.best_precision, o.best_recall
    (0.90909, 0.25, 0.83333, 1.0)

    >>> _scores = [0.1, 0.2, 0.3]
    >>> _labels = [0, 0, 0]
    >>> o = find_best_threshold(_scores, _labels)  # Labels are all negative, the threshold should be meaningless.
    >>> o.best_accuracy_threshold
    inf

    >>> _scores = [0.1, 0.2, 0.3]
    >>> _labels = [1, 1, 1]
    >>> o = find_best_threshold(_scores, _labels)  # Labels are all positive, the threshold should be meaningless.
    >>> o.best_accuracy_threshold
    -inf

    >>> _scores = [0.1, 0.2, 0.3]
    >>> _labels = [1, 1, 1]
    >>> o = find_best_threshold(_scores, _labels, greater_better=False)
    >>> o.best_accuracy_threshold
    inf

Args:
    scores: float array-like
    labels: 0/1 array-like
    greater_better: Default True, it means that 1 if greater than threshold, 0 otherwise;
        When False, it means that 0 if greater than threshold, 1 otherwise.
    epsilon:
    n_digits: round(f, n_digits)

```


### `BertTokenizer: Bert 分词器`
> [source](huaytools/nlp/bert/tokenization.py#L220)

```python
Bert 分词器

Examples:
    >>> text = '我爱python，我爱编程；I love python, I like programming. Some unkword'

    # WordPiece 切分
    >>> tokens = tokenizer.tokenize(text)
    >>> len(tokens)
    22
    >>> assert [tokens[2], tokens[-2], tokens[-7]] == ['python', '##nk', 'program']

    # 模型输入
    >>> tokens, token_ids, token_type_ids = tokenizer.encode(text, return_token_type_ids=True)
    >>> tokens[:6]
    ['[CLS]', '我', '爱', 'python', '，', '我']
    >>> assert token_ids[:6] == [101, 2769, 4263, 9030, 8024, 2769]
    >>> assert token_type_ids == [0] * len(token_ids)

    # 句对模式
    >>> txt1 = '我爱python'
    >>> txt2 = '我爱编程'
    >>> tokens, token_ids, token_masks = tokenizer.encode(txt1, txt2, return_token_masks=True)
    >>> tokens
    ['[CLS]', '我', '爱', 'python', '[SEP]', '我', '爱', '编', '程', '[SEP]']
    >>> assert token_ids == [101, 2769, 4263, 9030, 102, 2769, 4263, 5356, 4923, 102]
    >>> assert token_masks == [1] * 10

    >>> # batch 模式
    >>> ss = ['我爱python', '深度学习', '机器学习']
    >>> token_ids = tokenizer.batch_encode(ss)
    >>> len(token_ids), len(token_ids[0])
    (3, 6)

```


### `ner_result_parse: NER 结果解析（基于 BIO 格式）`
> [source](huaytools/nlp/ner_utils.py#L22)

```python
NER 结果解析（基于 BIO 格式）

Examples:
    >>> _label_id2name = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
    >>> _tokens = list('你知道小明生活在北京吗？')
    >>> _labels = list(map(int, '000120003400'))
    >>> ner_result_parse(_tokens, _labels, _label_id2name)
    [['PER', '小明', (3, 4)], ['LOC', '北京', (8, 9)]]

    >>> _tokens = list('小明生活在北京')  # 测试头尾是否正常
    >>> _labels = list(map(int, '1200034'))
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['PER', '小明', (0, 1)], ['LOC', '北京', (5, 6)]]

    >>> _tokens = list('明生活在北京')  # 明: I-PER
    >>> _labels = list(map(int, '200034'))
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['LOC', '北京', (4, 5)]]

    >>> _tokens = list('小明生活在北')
    >>> _labels = list(map(int, '120003'))  # 北: B-LOC
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['PER', '小明', (0, 1)], ['LOC', '北', (5, 5)]]

Args:
    tokens:
    labels:
    token_id2name:
    label_id2name:

Returns:
    example: [['小明', 'PER', (3, 4)], ['北京', 'LOC', (8, 9)]]
```


### `split: 将数据按比例切分`
> [source](huaytools/nlp/utils/_basic.py#L63)

```python
将数据按比例切分

Args:
    *arrays:
    split_size: 切分比例，采用向上取整：ceil(6*0.3) = 2
    random_seed: 随机数种子
    shuffled: 是否打乱

Examples:
    >>> data = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
    >>> xt, xv = split(*data, split_size=0.3, shuffled=False)
    >>> xt
    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    >>> xv
    [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    
Returns:
    x_train, x_val =  split(x)
    (a_train, b_train, c_train), (a_val, b_train, c_train) = split(a, b, c)
```


### `XLSHelper: Excel 文件加载（基于 openpyxl）`
> [source](huaytools/utils/excel_helper/_basic.py#L28)

```python
Excel 文件加载（基于 openpyxl）

Examples:
    >>> fp = r'./test_data.xlsx'
    >>> xh = XLSHelper(fp)
    >>> xh.get_data_from('Sheet2')
    [['S1', 'S2', 'S3'], ['a', 'b', 'c'], ['aa', 'bb', 'cc']]
    >>> type(xh.workbook)
    <class 'openpyxl.workbook.workbook.Workbook'>
    >>> list(xh.sheet_names)
    ['Sheet1', 'Sheet2']
    >>> xh.sheets['Sheet1']
    [['H1', 'H2', 'H3'], [1, 2, 3], [11, 22, 33]]
    >>> xh.sheets['Sheet2']
    [['S1', 'S2', 'S3'], ['a', 'b', 'c'], ['aa', 'bb', 'cc']]
    >>> xh.first_sheet
    [['H1', 'H2', 'H3'], [1, 2, 3], [11, 22, 33]]
    >>> xh.active_sheet
    [['S1', 'S2', 'S3'], ['a', 'b', 'c'], ['aa', 'bb', 'cc']]
```


### `ImageCheck: 图片完整性检查`
> [source](huaytools/vision/image_check.py#L21)

```python
图片完整性检查

Examples:
    >>> img = r'./_test_data/pok.jpg'
    >>> ImageCheck.is_complete(img)

```


### `get_real_ext: 获取图像文件的真实后缀`
> [source](huaytools/vision/image_utils.py#L21)

```python
获取图像文件的真实后缀
如果不是图片，返回后缀为 None
该方法不能判断图片是否完整

Args:
    image_path:
    return_is_same: 是否返回 `is_same`

Returns:
    ext_real, is_same
    真实后缀，真实后缀与当前后缀是否相同
    如果当前文件不是图片，则 ext_real 为 None
```


## Pytorch Models

### `DualNet: 双塔结构`
> [source](huaytools/pytorch/modules/advance/dual.py#L25)

```python
双塔结构
```


### `SiameseNet: 孪生网络，基于双塔结构`
> [source](huaytools/pytorch/modules/advance/siamese.py#L27)

```python
孪生网络，基于双塔结构
```


### `SimCSE: SimCSE`
> [source](huaytools/pytorch/modules/advance/sim_cse.py#L30)

```python
SimCSE

References: https://github.com/princeton-nlp/SimCSE
```


### `Bert: Bert by Pytorch`
> [source](huaytools/pytorch/nn/bert/_bert.py#L89)

```python
Bert by Pytorch

Examples:
    >>> bert = Bert()

    >>> ex_token_ids = torch.randint(100, [2, 3])
    >>> o = bert(ex_token_ids)
    >>> o[0].shape
    torch.Size([2, 768])
    >>> o[1].shape
    torch.Size([2, 3, 768])

    # Tracing
    >>> _ = bert.eval()  # avoid TracerWarning
    >>> traced_bert = torch.jit.trace(bert, (ex_token_ids,))
    >>> inputs = torch.randint(100, [5, 6])
    >>> torch.equal(traced_bert(inputs)[1], bert(inputs)[1])
    True

    # >>> print(traced_bert.code)

```


### `BertPretrain: Bert 预训练（MLM + NSP）`
> [source](huaytools/pytorch/nn/bert/_bert.py#L335)

```python
Bert 预训练（MLM + NSP）

References:
    https://github.com/microsoft/unilm/blob/master/unilm-v1/src/pytorch_pretrained_bert/modeling.py
    - BertForPreTraining
        - BertPreTrainingHeads
            - BertLMPredictionHead
```


### `SentenceBert: Bert 句向量`
> [source](huaytools/pytorch/nn/bert/bert_for_sentence_embedding.py#L31)

```python
Bert 句向量

References:
    [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
```


### `BertSequenceTagging: Bert 序列标注`
> [source](huaytools/pytorch/nn/bert/bert_for_sequence_tagging.py#L31)

```python
Bert 序列标注
```


### `BertTextClassification: Bert 文本分类`
> [source](huaytools/pytorch/nn/bert/bert_for_text_classification.py#L28)

```python
Bert 文本分类
```


### `LayerNorm: Layer Normalization`
> [source](huaytools/pytorch/nn/normalization/layer_norm.py#L28)

```python
Layer Normalization

Almost same as `nn.LayerNorm`
```


## Pytorch Utils

### `DictTensorDataset: 字典格式的 Dataset`
> [source](huaytools/pytorch/data/_basic.py#L34)

```python
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
```


### `ToyDataLoader: 简化创建 DataLoader 的过程`
> [source](huaytools/pytorch/data/_basic.py#L95)

```python
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
```


### `BertDataLoader: 简化 Bert 训练数据的加载`
> [source](huaytools/pytorch/data/data_loader_for_bert.py#L60)

```python
简化 Bert 训练数据的加载

Examples:
    # 单句判断
    >>> file = ['我爱Python', '我爱机器学习', '我爱NLP']
    >>> ds = []
    >>> for t in file:
    ...     it = BertSample(t)
    ...     ds.append(it)
    >>> dl = BertDataLoader(ds, batch_size=2)
    >>> first_batch = next(iter(dl))
    >>> first_batch['token_ids'].shape
    torch.Size([2, 8])
    >>> first_batch['token_ids'][0, :]  # 我爱Python
    tensor([ 101, 2769, 4263, 9030,  102,    0,    0,    0])
    >>> first_batch['token_ids'][1, :]  # 我爱机器学习
    tensor([ 101, 2769, 4263, 3322, 1690, 2110,  739,  102])

    # 句间关系
    >>> file = [('我爱Python', '测试1'), ('我爱机器学习', '测试2'), ('我爱NLP', '测试3')]
    >>> ds = [BertSample(t[0], t[1], label=1) for t in file]
    >>> dl = BertDataLoader(ds, batch_size=2)
    >>> for b in dl:
    ...     features, labels = b
    ...     print('max_len:', features['token_ids'].shape[1])
    ...     print('token_ids:', features['token_ids'][0, :10])
    ...     print('labels:', labels)
    ...     print()
    max_len: 12
    token_ids: tensor([ 101, 2769, 4263, 9030,  102, 3844, 6407,  122,  102,    0])
    labels: tensor([1., 1.])
    <BLANKLINE>
    max_len: 10
    token_ids: tensor([  101,  2769,  4263,   156, 10986,   102,  3844,  6407,   124,   102])
    labels: tensor([1.])
    <BLANKLINE>

    # 双塔
    >>> file = [('我爱Python', '测试1'), ('我爱机器学习', '测试2'), ('我爱NLP', '测试3')]
    >>> ds = [MultiBertSample(list(t)) for t in file]
    >>> dl = BertDataLoader(ds, batch_size=2)
    >>> first_batch = next(iter(dl))
    >>> len(first_batch)
    2
    >>> [it['token_ids'].shape for it in first_batch]  # noqa
    [torch.Size([2, 8]), torch.Size([2, 5])]

    # 多塔
    >>> file = [('我爱Python', '测试1', '1'), ('我爱机器学习', '测试2', '2'), ('我爱NLP', '测试3', '3')]
    >>> ds = [MultiBertSample(list(t)) for t in file]
    >>> dl = BertDataLoader(ds, batch_size=2)
    >>> first_batch = next(iter(dl))
    >>> len(first_batch)
    3
    >>> [it['token_ids'].shape for it in first_batch]  # noqa
    [torch.Size([2, 8]), torch.Size([2, 5]), torch.Size([2, 3])]

    # 异常测试
    >>> ds = ['我爱自然语言处理', '我爱机器学习', '测试']
    >>> dl = BertDataLoader(ds, batch_size=2)  # noqa
    Traceback (most recent call last):
        ...
    TypeError: Unsupported sample type=<class 'str'>

References:
    sentence_transformers.SentenceTransformer.smart_batching_collate
```


### `ContrastiveLoss: 对比损失（默认距离函数为欧几里得距离）`
> [source](huaytools/pytorch/modules/loss/contrastive.py#L49)

```python
对比损失（默认距离函数为欧几里得距离）
```


### `CrossEntropyLoss: 交叉熵`
> [source](huaytools/pytorch/modules/loss/cross_entropy.py#L214)

```python
交叉熵

TODO: 实现 weighted、smooth

Examples:
    >>> logits = torch.rand(5, 5)
    >>> labels = torch.arange(5)
    >>> probs = torch.softmax(logits, dim=-1)
    >>> onehot_labels = F.one_hot(labels)
    >>> my_ce = CrossEntropyLoss(reduction='none', onehot_label=True)
    >>> ce = nn.CrossEntropyLoss(reduction='none')
    >>> assert torch.allclose(my_ce(probs, onehot_labels), ce(logits, labels), atol=1e-5)

```


### `TripletLoss: Triplet 损失，常用于无监督学习、few-shot 学习`
> [source](huaytools/pytorch/modules/loss/triplet.py#L77)

```python
Triplet 损失，常用于无监督学习、few-shot 学习

Examples:
    >>> anchor = torch.randn(100, 128)
    >>> positive = torch.randn(100, 128)
    >>> negative = torch.randn(100, 128)

    # my_tl 默认 euclidean_distance_nosqrt
    >>> tl = TripletLoss(margin=2., reduction='none')
    >>> tld = nn.TripletMarginWithDistanceLoss(distance_function=euclidean_distance_nosqrt,
    ...                                        margin=2., reduction='none')
    >>> assert torch.allclose(tl(anchor, positive, negative), tld(anchor, positive, negative), atol=1e-5)

    # 自定义距离函数
    >>> from huaytools.pytorch.backend.distance_fn import cosine_distance
    >>> my_tl = TripletLoss(distance_fn=cosine_distance, margin=0.5, reduction='none')
    >>> tl = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5, reduction='none')
    >>> assert torch.allclose(my_tl(anchor, positive, negative), tl(anchor, positive, negative), atol=1e-5)

```


### `FGM: Fast Gradient Method (对抗训练)`
> [source](huaytools/pytorch/nn/adversarial_training/fast_gradient_method.py#L28)

```python
Fast Gradient Method (对抗训练)

Examples:
    >>> def training_step(model, batch, optimizer, fgm=FGM(param_pattern='word_embedding')):
    ...     inputs, labels = batch
    ...
    ...     # 正常训练
    ...     loss = model(inputs, labels)
    ...     loss.backward()  # 反向传播，得到正常的梯度
    ...
    ...     # 对抗训练（需要添加的代码）
    ...     fgm.collect(model)
    ...     fgm.attack()
    ...     loss_adv = model(inputs, labels)  # 对抗梯度
    ...     loss_adv.backward()  # 累计对抗梯度
    ...     fgm.restore(model)  # 恢复被添加扰动的参数
    ...
    ...     # 更新参数
    ...     optimizer.step()
    ...     optimizer.zero_grad()

References:
    - [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)
    - [NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
```


### `PGM: Projected Gradient Method (对抗训练)`
> [source](huaytools/pytorch/nn/adversarial_training/projected_gradient_descent.py#L28)

```python
Projected Gradient Method (对抗训练)

Examples:
    >>> def training_step(model, batch, optimizer, steps=3, pgm=PGM(param_pattern='word_embedding')):
    ...     inputs, labels = batch
    ...
    ...     # 正常训练
    ...     loss = model(inputs, labels)
    ...     loss.backward()  # 反向传播，得到正常的梯度
    ...
    ...     # 对抗训练（需要添加的代码）
    ...     pgm.collect(model)
    ...     for t in range(steps):
    ...         pgm.attack()  # 小步添加扰动
    ...
    ...         if t < steps - 1:
    ...             optimizer.zero_grad()  # 在最后一步前，还没有得到最终对抗训练的梯度，所以要先清零
    ...         else:
    ...             pgm.restore_grad(model)  # 最后一步时恢复正常的梯度，与累积的扰动梯度合并
    ...
    ...         loss_adv = model(inputs, labels)
    ...         loss_adv.backward()  # 累加对抗梯度（在最后一步之前，实际只有对抗梯度）
    ...     pgm.restore(model)  # 恢复被添加扰动的参数
    ...
    ...     # 更新参数
    ...     optimizer.step()
    ...     optimizer.zero_grad()

References:
    - [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
    - [NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
```


### `Trainer: Trainer 基类`
> [source](huaytools/pytorch/train/trainer.py#L53)

```python
Trainer 基类

Examples:
    # See examples/pytorch_trainer/*

Note: （约定）TODO
    1.
```


### `set_seed: 设置全局随机数种子，使实验可复现`
> [source](huaytools/pytorch/utils/_basic.py#L45)

```python
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
```


### `init_weights: 默认参数初始化`
> [source](huaytools/pytorch/utils/_basic.py#L156)

```python
默认参数初始化

Examples:
    >>> model = nn.Transformer()
    >>> _ = model.apply(init_weights)

Args:
    module:
    normal_std:

References: Bert
```


### `mixup: mixup 数据增强策略`
> [source](huaytools/pytorch/utils/mixup.py#L31)

```python
mixup 数据增强策略

Args:
    x:
    y:
    a:

Examples:
    >>> x = torch.randn(3, 5)
    >>> y = F.one_hot(torch.arange(3)).to(torch.float32)
    >>> x_, y_ = mixup(x, y, 0.2)

    ```python
    # How to use mixup in model.
    def forward(self, x, target=None, use_mixup=False, mixup_alpha=None):
        x = self.layer1(x)

        if use_mixup and self.training:
            x, target = mixup(x, target, mixup_alpha)

        x = self.layer2(x)

        if self.training:
            return x, target
        else:
            return x
    ```

References:
    https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/models/utils.py
    - mixup_process
```


## Python Utils

### `simple_argparse: 一个简化版 argparse`
> [source](huaytools/python/custom/simple_argparse.py#L28)

```python
一个简化版 argparse

不需要预先设置字段，严格按照 `--a A` 一组的方式自动提取，
    其中 A 部分会调用 eval()，某种程度上比自带的 argparse 更强大

Examples:
    >>> sys.argv = ['xxx.py', '--a', 'A', '--b', '1', '--c', '3.14', '--d', '[1,2]', '--e', '"[1,2]"']
    >>> simple_argparse()
    {'a': 'A', 'b': 1, 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> _args = BunchDict(x=1, b=20)
    >>> simple_argparse(_args)
    {'x': 1, 'b': 1, 'a': 'A', 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> sys.argv = ['xxx.py']
    >>> simple_argparse(_args)
    {'x': 1, 'b': 1, 'a': 'A', 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> sys.argv = ['xxx.py', '-a', 'A']
    >>> simple_argparse()
    Traceback (most recent call last):
        ...
    AssertionError: `-a` should starts with "--"

```


### `ArrayDict: 数组字典，支持 slice`
> [source](huaytools/python/custom/special_dict.py#L37)

```python
数组字典，支持 slice

Examples:
    >>> d = ArrayDict(a=1, b=2)
    >>> d
    ArrayDict([('a', 1), ('b', 2)])
    >>> d['a']
    1
    >>> d[1]
    ArrayDict([('b', 2)])
    >>> d['c'] = 3
    >>> d[0] = 100
    Traceback (most recent call last):
        ...
    TypeError: ArrayDict cannot use `int` as key.
    >>> d[1: 3]
    ArrayDict([('b', 2), ('c', 3)])
    >>> print(*d)
    a b c
    >>> d.setdefault('d', 4)
    4
    >>> print(d)
    ArrayDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    >>> d.pop('a')
    1
    >>> d.update({'b': 20, 'c': 30})
    >>> def f(**d): print(d)
    >>> f(**d)
    {'b': 20, 'c': 30, 'd': 4}

```


### `ValueArrayDict: 数组字典，支持 slice，且操作 values`
> [source](huaytools/python/custom/special_dict.py#L98)

```python
数组字典，支持 slice，且操作 values

Examples:
    >>> d = ValueArrayDict(a=1, b=2)
    >>> d
    ValueArrayDict([('a', 1), ('b', 2)])
    >>> assert d[1] == 2
    >>> d['c'] = 3
    >>> assert d[2] == 3
    >>> d[1:]
    (2, 3)
    >>> print(*d)  # 注意打印的是 values
    1 2 3
    >>> del d['a']
    >>> d.update({'a':10, 'b': 20})
    >>> d
    ValueArrayDict([('b', 20), ('c', 3), ('a', 10)])

```


### `BunchDict: 基于 dict 实现 Bunch 模式`
> [source](huaytools/python/custom/special_dict.py#L164)

```python
基于 dict 实现 Bunch 模式

Examples:
    # 直接使用
    >>> d = BunchDict(a=1, b=2)
    >>> d
    {'a': 1, 'b': 2}
    >>> d.c = 3
    >>> assert 'c' in d and d.c == 3
    >>> dir(d)
    ['a', 'b', 'c']
    >>> assert 'a' in d
    >>> del d.a
    >>> assert 'a' not in d
    >>> d.dict
    {'b': 2, 'c': 3}

    # 从字典加载
    >>> x = {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}
    >>> y = BunchDict.from_dict(x)
    >>> y
    {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}

    # 预定义配置
    >>> class Config(BunchDict):
    ...     def __init__(self, **config_items):
    ...         from datetime import datetime
    ...         self.a = 1
    ...         self.b = 2
    ...         self.c = datetime(2012, 1, 1)  # 注意是一个特殊对象，默认 json 是不支持的
    ...         super().__init__(**config_items)
    >>> args = Config(b=20)
    >>> args.a = 10
    >>> args
    {'a': 10, 'b': 20, 'c': datetime.datetime(2012, 1, 1, 0, 0)}
    >>> args == args.dict
    True
    >>> # 添加默认中不存的配置项
    >>> args.d = 40
    >>> print(args.get_pretty_dict())  # 注意 'b' 保存成了特殊形式
    {
        "a": 10,
        "b": 20,
        "c": "datetime.datetime(2012, 1, 1, 0, 0)__@AnyEncoder@__gASVKgAAAAAAAACMCGRhdGV0aW1llIwIZGF0ZXRpbWWUk5...",
        "d": 40
    }

    # 保存/加载
    >>> fp = r'./-test/test_save_config.json'
    >>> os.makedirs(os.path.dirname(fp), exist_ok=True)
    >>> args.save(fp)  # 保存
    >>> x = Config.load(fp)  # 重新加载
    >>> assert x == args.dict
    >>> _ = os.system('rm -rf ./-test')

References:
    - bunch（pip install bunch）
```


### `FieldBunchDict: 基于 dataclass 的 BunchDict`
> [source](huaytools/python/custom/special_dict.py#L303)

```python
基于 dataclass 的 BunchDict

原来预定义的参数，需要写在 __init__ 中：
    ```
    class Args(BunchDict):
        def __init__(self):
            a = 1
            b = 2
    ```
现在可以直接当作 dataclass 来写：
    ```
    @dataclass()
    class Args(BunchDict):
        a: int = 1
        b: int = 2
    ```

Examples:
    # 预定义配置
    >>> @dataclass()
    ... class Config(FieldBunchDict):
    ...     from datetime import datetime
    ...     a: int = 1
    ...     b: int = 2
    ...     c: Any = datetime(2012, 1, 1)  # 注意是一个特殊对象，默认 json 是不支持的
    >>> args = Config(b=20)
    >>> args.a = 10
    >>> args
    Config(a=10, b=20, c=datetime.datetime(2012, 1, 1, 0, 0))
    >>> args.dict
    {'a': 1, 'b': 20, 'c': datetime.datetime(2012, 1, 1, 0, 0)}
    >>> args.d = 40  # 默认中没有的配置项（不推荐，建议都定义在继承类中，并设置默认值）
    Traceback (most recent call last):
        ...
    KeyError: '`d` not in fields. If it has to add new field, recommend to use `BunchDict`'

    # 保存/加载
    >>> fp = r'./-test/test_save_config.json'
    >>> os.makedirs(os.path.dirname(fp), exist_ok=True)
    >>> args.save(fp)  # 保存
    >>> x = Config.load(fp)  # 重新加载
    >>> assert x == args.dict
    >>> _ = os.system('rm -rf ./-test')

```


### `ls_dir_recur: 递归遍历目录下的所有文件`
> [source](huaytools/python/file_utils.py#L24)

```python
递归遍历目录下的所有文件

Args:
    src_path:
    cond_fn: 条件函数，传入文件完整路径，判断是否加入返回列表
```


### `files_concat: 文件拼接`
> [source](huaytools/python/file_utils.py#L47)

```python
文件拼接

Examples:
    >>> _dir = r'./-test'
    >>> os.makedirs(_dir, exist_ok=True)
    >>> f1 = os.path.join(_dir, r't1.txt')
    >>> os.system(f'echo 123 > {f1}')
    0
    >>> f2 = '456'  # f2 = os.path.join(_dir, r't2.txt')
    >>> _out = files_concat([f1, f2])  # 可以拼接文件、字符串
    >>> print(_out)
    123
    456
    <BLANKLINE>
    >>> _out = files_concat([f1, f2], '---')
    >>> print(_out)
    123
    ---
    456
    <BLANKLINE>
    >>> os.system(f'rm -rf {_dir}')
    0

```


### `get_caller_name: 获取调用者的名称`
> [source](huaytools/python/utils/_basic.py#L53)

```python
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

```


### `function_test_dn: 函数测试装饰器`
> [source](huaytools/python/utils/_basic.py#L331)

```python
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
```
