提示学习（Prompt Learning）
===

- [PET: Pattern-Exploiting Training](#pet-pattern-exploiting-training)
- [P-tuning](#p-tuning)
- [参考资料](#参考资料)

## PET: Pattern-Exploiting Training
> $[1]$、$[2]$

利用**掩码语言模型**（Masked Language Model, MLM），将一般分类任务转化为“完形填空”任务；  

简单来说，就在原始文本上添加一段**描述性语句**，并将描述中可以**作为分类依据的关键词** Mask，然后利用 MLM 预测被 Mask 掉的部分，进而得到原始文本的分类结果  

**示例**
```
分类任务：
    原始文本：  “八个月了，终于又能在赛场上看到女排姑娘们了。”
    拓展后：    “下面报导一则____新闻。八个月了，终于又能在赛场上看到女排姑娘们了。”  
    Mask 内容：体育

NLI 任务
    “我去了北京？____，我去了上海。”  -> 不是
    “我去了北京？____，我在天安门。”  -> 是的
```

组合后的语句应该保持尽可能通顺自然，不能过于生硬。否则可能就退化成了一般的分类任务。

PET 流程的**半监督**用法[$^{[1]}$](#ref1)：
```
同一任务可以使用多种不同的 Pattern；

1. 对每种 Pattern，单独训练一个 MLM；
2. 使用集成模型预测未标注数据，得到伪标签；
3. 用所有伪标签数据训练一个常规的分类模型
```

PET 直接分类的用法：

## P-tuning
> 


## 参考资料
<a name="ref1"> $[1]$ </a> [[2001.07676] Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)
> 提出 Pattern-Exploiting Training

$[2]$ [[2009.07118] It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118)  
> 拓展 Pattern-Exploiting Training

$[3]$ [必须要GPT3吗？不，BERT的MLM模型也能小样本学习 - 科学空间|Scientific Spaces](https://kexue.fm/archives/7764)
> 介绍 Pattern-Exploiting Training