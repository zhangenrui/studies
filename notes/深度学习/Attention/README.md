Attention
===

- [Multi-head Self Attention](#multi-head-self-attention)
    - [前向过程](#前向过程)
    - [伪代码](#伪代码)

## Multi-head Self Attention

### 前向过程

$$
\begin{aligned}
    \text{Attention}(Q,K,V) &= \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V \\
    \text{head}_\text{i} &= \text{Attention}(QW_i^Q,KW_i^K,VW_i^V) \\
    \text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,..,\text{head}_\text{h})W^O
\end{aligned}
$$

### 伪代码
> 合并处理 Multi-head

```python
# 输入
q = k = v = x  # [B, T, H*N]

# linear
q = linear_q(q).reshape([B, T, H, N]).transpose(1, 2)  # [B, H, T, N]
k = linear_k(k).reshape([B, S, H, N]).transpose(1, 2)  # [B, H, T, N]
v = linear_v(v).reshape([B, S, H, N]).transpose(1, 2)  # [B, H, T, N]

# attention
logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
attention_score = softmax(logits)

# output
o = torch.matmul(score, v)
```