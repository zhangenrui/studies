预训练语言模型
===

- [前言](#前言)
    - [为什么需要预训练语言模型（PLM）](#为什么需要预训练语言模型plm)
    - [PLM 的发展历史](#plm-的发展历史)
    - [PLM 分类体系](#plm-分类体系)
- [参考资料](#参考资料)

## 前言

- 预训练语言模型（Pretrained Language Model, PLM）

### 为什么需要预训练语言模型（PLM）

- 预训练属于迁移学习的范畴；预训练好的模型相当于自带一套“先验知识”；
- 预训练的基本思想：先在一些上游任务上训练好一组模型参数，然后在目标任务上基于这组参数进行微调（Fine tune）；
- 视觉领域预训练模型的成果；

### PLM 的发展历史

- word2vec/GloVe
- Attention/Self-Attention
- Transformer
- BERT、GPT、ELMo

### PLM 分类体系

- 是否上下文相关
    - 上下文无关
        - Word2Vec(CBOW、Skip-gram)、GloVe
    - 上下文相关
        - BERT、GPT、ELMo、...
- 核心结构
    - LSTM
        - ELMo、CoVe、...
    - Transformer Encoder
        - BERT、SpanBERT、RoBERTa、XLNet、...
    - Transformer Decoder
        - GPT系列
    - Transformer
        - MASS、BART、XNLG、mBART
- 任务类型
    - 有监督
        - 机器翻译
            - CoVe
    - 无监督/自监督
        - 语言模型
            - ELMo、GTP系列、UniLM
        - 掩码语言模型
            - BERT、
        - 排列语言模型
            - XLNet
        - 降噪自编码
            - BART
        - 对比学习
            - 替换符检测
            - 下一句预测
            - 语序预测

## 参考资料
$[1]$ 《预训练语言模型》邵浩、刘一烽编著