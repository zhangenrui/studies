RNN
===

- [RNN](#rnn)
    - [RNN 的前向过程](#rnn-的前向过程)
- [LSTM](#lstm)
    - [LSTM 的前向过程](#lstm-的前向过程)
    - [常见问题](#常见问题)
- [GRU](#gru)
    - [GRU 的前向过程](#gru-的前向过程)
    - [常见问题](#常见问题-1)

## RNN

### RNN 的前向过程

$$
\begin{aligned}
    y_t &= W[h_{t-1},x_t] + b \\ 
    h_t &= \tanh(a_t) 
\end{aligned}
$$
> 

或

$$
\begin{aligned}
    y_t &= W[y_{t-1},x_t] + b \\ 
    h_t &= \tanh(a_t) 
\end{aligned}
$$
> $[x1,x2]$ 表示**向量拼接**；RNN 为递推结构，其中 $h_0$ 一般初始化为 0；

> 前者来自 Elman，后者来自 Jordan；两个过程的区别仅在于当前步的输入不同，一般所说的 RNN 指的是前者；
>> [Recurrent neural network - Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)


## LSTM

### LSTM 的前向过程

$$
\begin{aligned}
    f_t &= \sigma(W_f[h_{t-1},x_t] + b_t) \\
    i_t &= \sigma(W_i[H_{t-1},x_t] + b_i) \\
    \tilde{C}_t &= \tanh(W_C[h_{t-1},x_t] + b_C) \\
    C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \\
    o_t &= \sigma(W_o[h_{t-1},x_t] + b_o) \\
    h_t &= o_t * \tanh(C_t)
\end{aligned}
$$
> $[x1,x2]$ 表示**向量拼接**；$*$ 表示**按位相乘**；

### 常见问题

- **LSTM 是如何实现长短期记忆的？**
    - 利用“遗忘门”和“输入门”控制“长期记忆”和“短期记忆”的权重，从而实现长短期记忆（具体参考前向传播公式）；
    - 如果长期记忆比较重要，那么“遗忘门”输出的权重较高；反之短期记忆比较重要时，就是“记忆门”的权重更高；
- **LSTM 中各个门的作用是什么？**
    - “**遗忘门**”控制前一步记忆状态中的信息有多大程度被遗忘；
    - “**输入门**（记忆门）”控制当前计算的新状态以多大的程度更新到记忆状态中；
    - “**输出门**”控制当前的输出有多大程度取决于当前的记忆状态；
- **LSTM 前向过程（门的顺序）**
    - 遗忘门 -> 输入门 -> 输出门


## GRU

### GRU 的前向过程

$$
\begin{aligned}
    z_t &= \sigma(W_z[h_{t-1},x_t] + b_z) \\
    r_t &= \sigma(W_r[h_{t-1},x_t] + b_r) \\
    \tilde{h}_t &= \tanh(W[r_t*h_{t-1},x_t] + b) \\
    h_t &= (1-z_t)*h_{t-1} + z_t * \tilde{h}_t
\end{aligned}
$$
> $[x1,x2]$ 表示**向量拼接**；$*$ 表示**按位相乘**；

### 常见问题

- **GRU 中门的作用**
    - “更新门”用于控制前一时刻的状态信息被融合到当前状态中的程度；
    - “重置门”用于控制忽略前一时刻的状态信息的程度
- **与 LSTM 的区别**
    - 将 LSTM 中的“遗忘门”和“输入门”合并为“更新门”；
    - “重置门”代替“输出门”；