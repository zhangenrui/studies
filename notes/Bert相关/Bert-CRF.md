
### BERT 在序列标注任务中为什么没有用到 CRF？
> [BERT中进行NER为什么没有使用CRF，CRF在序列标注中是必备的么？ - 知乎](https://www.zhihu.com/question/358892919)

**小结**
- CRF 在序列标注中不是必备的；
- BERT+Softmax 的拟合能力已经足够强，不需要 CRF；

**参考**

<details><summary><b>transformers.models.bert.BertForTokenClassification 的实现</b></summary>

```python
# 代码有省略
class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, token_ids, masks, labels):
        sequence_output = self.bert(token_ids, masks)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            if masks is not None:
                active_loss = masks.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
                )
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            return logits, loss

        return logits
```

</details>

### 相关问题

#### 序列标注任务中使用 CRF 的目的是什么？

- 利用学习到的转移概率，（尽量）保证相邻标签的状态转移符合约定（比如 `B->I` 是合法的，而 `O->I` 是非法的）；
    > 基于这个目的，甚至可以手动配置转移矩阵，比如把合法的转移值设为统计概率，不合法的设为负无穷；

#### 为什么使用 BERT+CRF 的效果不明显？
> [你的CRF层的学习率可能不够大 - 科学空间|Scientific Spaces](https://kexue.fm/archives/7196)  

- 绝大多数情况下，使用的 BERT 都是经过预训练的，在下游任务中进行 fine tune 时的学习率一般在 `1e-5` 的级别；而此时 CRF 的参数还是随机初始化的，在如此小的学习率下，很难学习到正确的转移概率，导致效果不佳；

#### 如果用 BERT+CRF 对下游任务做微调，有什么需要注意的？

- 单独为 CRF 层设置一个较大的学习率，如 `1e-2 ~ 1e-3`；
