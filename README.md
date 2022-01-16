studies
===

![python-version](https://img.shields.io/badge/python-3.8+-green)
![pytorch-version](https://img.shields.io/badge/pytorch-1.8+-green)
![tf-version](https://img.shields.io/badge/tensorflow-2.3+-green)
![repo-size](https://img.shields.io/github/repo-size/imhuay/studies)
<!-- ![total-lines](https://img.shields.io/tokei/lines/github/imhuay/studies) -->
<!-- ![code-size](https://img.shields.io/github/languages/code-size/imhuay/studies) -->

<!-- ![followers](https://img.shields.io/github/followers/imhuay?style=social) -->
<!-- ![user-stars](https://img.shields.io/github/stars/imhuay?style=social) -->


### TODO

- [ ] 自动生成目录结构（books、papers 等）
<!-- - [ ] 【`2021.11.11`】pytorch_trainer: 为 EvaluateCallback 添加各种预定义评估指标，如 acc、f1 等，目前只有 loss； -->
<!-- - [ ] 【`2021.11.11`】论文：What does BERT learn about the structure of language? —— Bert 各层的含义； -->
<!-- - [ ] 【`2021.11.10`】bert-tokenizer 自动识别 `[MASK]` 等特殊标识； -->
<!-- - [ ] 【`2021.11.07`】面试笔记：通识问题/项目问题 -->
<!-- - [ ] 【`2021.10.22`】max_batch_size 估算 -->

<details><summary><b> Done </b></summary>

- [x] 【`2021.11.12`】优化 auto-readme，使用上一次的 commit info，而不是默认 'Auto-README'
    - 参考：`git commit -m "$(git log -"$(git rev-list origin/master..master --count)" --pretty=%B | cat)"`
    - 说明：使用 origin/master 到 master 之间所有的 commit 信息作为这次的 message；
- [x] 【`2021.11.11`】bert 支持加载指定层 -> `_test_load_appointed_layers()`
- [x] 【`2021.11.08`】把 __test.py 文件自动加入文档测试（放弃）
    - 有些测试比较耗时，不需要全部加入自动测试；
    - __test.py 针对的是存在相对引用的模块，如果这些模块有改动，会即时测试，所以也不需要自动测试
- [x] 【`2021.11.03`】[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 代码阅读

</details>

### 其他仓库
- [Algorithm_Interview_Notes-Chinese](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese_backups): 在校期间的学习/面试笔记；
- [bert_by_keras](https://github.com/imhuay/bert_by_keras): 使用 keras 重构的 Bert；
- [algorithm](https://github.com/imhuay/algorithm): 刷题笔记，实际上就是本仓库 algorithm 目录下的内容；

---

<font color="LightGrey"><i> `The following is Auto-generated` </i></font>

---

Repo Index
---

- [Algorithm Studies](#algorithm-studies)
- [Coding Lab](#coding-lab)

---

Algorithm Studies
---

<details><summary><b> 基础-模板 [2] <a href="algorithms/topics/基础-模板.md">¶</a></b></summary>

- [`+模板 字符串 split切分 (简单, 模板库-基础)`](algorithms/topics/基础-模板.md#模板-字符串-split切分-简单-模板库-基础)
- [`+模板 搜索 二分查找 (简单, 模板库-基础)`](algorithms/topics/基础-模板.md#模板-搜索-二分查找-简单-模板库-基础)

</details>

<details><summary><b> 基础-经典问题&代码 [18] <a href="algorithms/topics/基础-经典问题&代码.md">¶</a></b></summary>

- [`LeetCode No.0072 编辑距离 (困难, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#leetcode-no0072-编辑距离-困难-2022-01)
- [`剑指Offer No.007 重建二叉树 (中等, 2021-11)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no007-重建二叉树-中等-2021-11)
- [`剑指Offer No.016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.024 反转链表 (简单, 2021-11)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no024-反转链表-简单-2021-11)
- [`剑指Offer No.029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.049 丑数 (中等, 2021-12)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no049-丑数-中等-2021-12)
- [`剑指Offer No.051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no051-数组中的逆序对-困难-2022-01)
- [`剑指Offer No.060 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no060-n个骰子的点数-中等-2022-01)
- [`剑指Offer No.062 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no062-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer No.067 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no067-把字符串转换成整数atoi-中等-2022-01)
- [`剑指Offer No.068 1-二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/基础-经典问题&代码.md#剑指offer-no068-1-二叉搜索树的最近公共祖先-简单-2022-01)
- [`剑指Offer2 No.001 整数除法 (中等, 2022-02)`](algorithms/topics/基础-经典问题&代码.md#剑指offer2-no001-整数除法-中等-2022-02)

</details>

<details><summary><b> 题集-LeetCode [27] <a href="algorithms/topics/题集-LeetCode.md">¶</a></b></summary>

- [`LeetCode No.0001 两数之和 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0001-两数之和-简单-2021-10)
- [`LeetCode No.0002 两数相加 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0002-两数相加-中等-2021-10)
- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0011 盛最多水的容器 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0011-盛最多水的容器-中等-2021-10)
- [`LeetCode No.0015 三数之和 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0015-三数之和-中等-2021-10)
- [`LeetCode No.0016 最接近的三数之和 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0016-最接近的三数之和-中等-2021-10)
- [`LeetCode No.0021 合并两个有序链表 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0021-合并两个有序链表-简单-2021-10)
- [`LeetCode No.0029 两数相除 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0029-两数相除-中等-2021-10)
- [`LeetCode No.0033 搜索旋转排序数组 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0033-搜索旋转排序数组-中等-2021-10)
- [`LeetCode No.0042 接雨水 (困难, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0042-接雨水-困难-2021-10)
- [`LeetCode No.0053 最大子数组和 (简单, 2022-01)`](algorithms/topics/题集-LeetCode.md#leetcode-no0053-最大子数组和-简单-2022-01)
- [`LeetCode No.0072 编辑距离 (困难, 2022-01)`](algorithms/topics/题集-LeetCode.md#leetcode-no0072-编辑距离-困难-2022-01)
- [`LeetCode No.0086 分隔链表 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0086-分隔链表-中等-2021-10)
- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`LeetCode No.0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode No.0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/题集-LeetCode.md#leetcode-no0300-最长递增子序列-中等-2022-01)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/题集-LeetCode.md#leetcode-no0343-整数拆分-中等-2021-12)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0434-字符串中的单词数-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0437-路径总和3-中等-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`LeetCode No.0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/题集-LeetCode.md#leetcode-no0496-下一个更大元素-简单-2021-11)
- [`LeetCode No.0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0611-有效三角形的个数-中等-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/题集-LeetCode.md#leetcode-no0859-亲密字符串-简单-2021-11)

</details>

<details><summary><b> 题集-剑指Offer [75] <a href="algorithms/topics/题集-剑指Offer.md">¶</a></b></summary>

- [`剑指Offer No.003 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no003-数组中重复的数字-简单-2021-11)
- [`剑指Offer No.004 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no004-二维数组中的查找-中等-2021-11)
- [`剑指Offer No.005 替换空格 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no005-替换空格-简单-2021-11)
- [`剑指Offer No.006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.007 重建二叉树 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no007-重建二叉树-中等-2021-11)
- [`剑指Offer No.009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.010 斐波那契数列-2（青蛙跳台阶） (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no010-斐波那契数列-2青蛙跳台阶-简单-2021-11)
- [`剑指Offer No.011 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no011-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer No.012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.013 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no013-机器人的运动范围-中等-2021-11)
- [`剑指Offer No.014 1-剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no014-1-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer No.014 2-剪绳子 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no014-2-剪绳子-中等-2021-11)
- [`剑指Offer No.015 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no015-二进制中1的个数-简单-2021-11)
- [`剑指Offer No.016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.017 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no017-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer No.018 删除链表的节点 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no018-删除链表的节点-简单-2021-11)
- [`剑指Offer No.019 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no019-正则表达式匹配-困难-2021-11)
- [`剑指Offer No.020 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no020-表示数值的字符串-中等-2021-11)
- [`剑指Offer No.021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.024 反转链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no024-反转链表-简单-2021-11)
- [`剑指Offer No.025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.026 树的子结构 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no026-树的子结构-中等-2021-11)
- [`剑指Offer No.027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no032-3-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer No.033 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no033-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer No.034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.037 序列化二叉树 (困难, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no037-序列化二叉树-困难-2021-12)
- [`剑指Offer No.038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no041-数据流中的中位数-困难-2021-12)
- [`剑指Offer No.042 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no042-连续子数组的最大和-简单-2021-12)
- [`剑指Offer No.043 1～n整数中1出现的次数 (困难, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no043-1n整数中1出现的次数-困难-2021-12)
- [`剑指Offer No.044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.045 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no045-把数组排成最小的数-中等-2021-12)
- [`剑指Offer No.046 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no046-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer No.047 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no047-礼物的最大价值-中等-2021-12)
- [`剑指Offer No.048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.049 丑数 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no049-丑数-中等-2021-12)
- [`剑指Offer No.050 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no050-第一个只出现一次的字符-简单-2021-12)
- [`剑指Offer No.051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no051-数组中的逆序对-困难-2022-01)
- [`剑指Offer No.052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no052-两个链表的第一个公共节点-简单-2022-01)
- [`剑指Offer No.053 1-求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no053-1-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer No.053 2-在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no053-2-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer No.054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no054-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer No.055 1-求二叉树的深度 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no055-1-求二叉树的深度-简单-2022-01)
- [`剑指Offer No.055 2-判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no055-2-判断是否为平衡二叉树-简单-2022-01)
- [`剑指Offer No.056 1-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no056-1-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer No.056 2-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no056-2-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer No.057 1-和为s的两个数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no057-1-和为s的两个数字-简单-2022-01)
- [`剑指Offer No.057 2-和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no057-2-和为s的连续正数序列-简单-2022-01)
- [`剑指Offer No.058 1-翻转单词顺序 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no058-1-翻转单词顺序-简单-2022-01)
- [`剑指Offer No.058 2-左旋转字符串 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no058-2-左旋转字符串-简单-2022-01)
- [`剑指Offer No.059 1-滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no059-1-滑动窗口的最大值-困难-2022-01)
- [`剑指Offer No.059 2-队列的最大值 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no059-2-队列的最大值-中等-2022-01)
- [`剑指Offer No.060 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no060-n个骰子的点数-中等-2022-01)
- [`剑指Offer No.061 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no061-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer No.062 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no062-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer No.063 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no063-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer No.064 求1~n的和 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no064-求1n的和-中等-2022-01)
- [`剑指Offer No.065 不用加减乘除做加法 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no065-不用加减乘除做加法-简单-2022-01)
- [`剑指Offer No.066 构建乘积数组 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no066-构建乘积数组-中等-2022-01)
- [`剑指Offer No.067 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no067-把字符串转换成整数atoi-中等-2022-01)
- [`剑指Offer No.068 1-二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no068-1-二叉搜索树的最近公共祖先-简单-2022-01)
- [`剑指Offer No.068 2-二叉树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no068-2-二叉树的最近公共祖先-简单-2022-01)

</details>

<details><summary><b> 题集-剑指Offer2 [3] <a href="algorithms/topics/题集-剑指Offer2.md">¶</a></b></summary>

- [`剑指Offer2 No.001 整数除法 (中等, 2022-02)`](algorithms/topics/题集-剑指Offer2.md#剑指offer2-no001-整数除法-中等-2022-02)
- [`剑指Offer2 No.069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/题集-剑指Offer2.md#剑指offer2-no069-山峰数组的顶部-简单-2022-02)
- [`剑指Offer2 No.076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/题集-剑指Offer2.md#剑指offer2-no076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 题集-程序员面试金典 [2] <a href="algorithms/topics/题集-程序员面试金典.md">¶</a></b></summary>

- [`程序员面试金典 No.0101 判定字符是否唯一 (简单, 2022-01)`](algorithms/topics/题集-程序员面试金典.md#程序员面试金典-no0101-判定字符是否唯一-简单-2022-01)
- [`程序员面试金典 No.0102 判定是否互为字符重排 (简单, 2022-01)`](algorithms/topics/题集-程序员面试金典.md#程序员面试金典-no0102-判定是否互为字符重排-简单-2022-01)

</details>

<details><summary><b>更多细分类型 ...<a href="algorithms/README.md">¶</a></b></summary>

<details><summary><b> 基础-模拟、数学、找规律 [18] <a href="algorithms/topics/基础-模拟、数学、找规律.md">¶</a></b></summary>

- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0343-整数拆分-中等-2021-12)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0859-亲密字符串-简单-2021-11)
- [`剑指Offer No.014 1-剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no014-1-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer No.014 2-剪绳子 (中等, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no014-2-剪绳子-中等-2021-11)
- [`剑指Offer No.029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.043 1～n整数中1出现的次数 (困难, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no043-1n整数中1出现的次数-困难-2021-12)
- [`剑指Offer No.044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.060 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no060-n个骰子的点数-中等-2022-01)
- [`剑指Offer No.060 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no060-n个骰子的点数-中等-2022-01)
- [`剑指Offer No.061 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no061-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer No.062 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no062-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer No.063 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no063-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer No.067 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no067-把字符串转换成整数atoi-中等-2022-01)

</details>

<details><summary><b> 技巧-位运算 [6] <a href="algorithms/topics/技巧-位运算.md">¶</a></b></summary>

- [`LeetCode No.0029 两数相除 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-no0029-两数相除-中等-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`剑指Offer No.015 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/技巧-位运算.md#剑指offer-no015-二进制中1的个数-简单-2021-11)
- [`剑指Offer No.056 1-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-no056-1-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer No.056 2-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-no056-2-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer No.065 不用加减乘除做加法 (简单, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-no065-不用加减乘除做加法-简单-2022-01)

</details>

<details><summary><b> 技巧-前缀和 [2] <a href="algorithms/topics/技巧-前缀和.md">¶</a></b></summary>

- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/技巧-前缀和.md#leetcode-no0437-路径总和3-中等-2021-10)
- [`剑指Offer No.066 构建乘积数组 (中等, 2022-01)`](algorithms/topics/技巧-前缀和.md#剑指offer-no066-构建乘积数组-中等-2022-01)

</details>

<details><summary><b> 技巧-单调栈、单调队列 [2] <a href="algorithms/topics/技巧-单调栈、单调队列.md">¶</a></b></summary>

- [`LeetCode No.0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/技巧-单调栈、单调队列.md#leetcode-no0496-下一个更大元素-简单-2021-11)
- [`剑指Offer No.059 1-滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/技巧-单调栈、单调队列.md#剑指offer-no059-1-滑动窗口的最大值-困难-2022-01)

</details>

<details><summary><b> 技巧-双指针 [12] <a href="algorithms/topics/技巧-双指针.md">¶</a></b></summary>

- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0011 盛最多水的容器 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0011-盛最多水的容器-中等-2021-10)
- [`LeetCode No.0015 三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0015-三数之和-中等-2021-10)
- [`LeetCode No.0016 最接近的三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0016-最接近的三数之和-中等-2021-10)
- [`LeetCode No.0042 接雨水 (困难, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0042-接雨水-困难-2021-10)
- [`LeetCode No.0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode No.0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-no0611-有效三角形的个数-中等-2021-10)
- [`剑指Offer No.021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/技巧-双指针.md#剑指offer-no021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-双指针.md#剑指offer-no048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.057 1-和为s的两个数字 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-no057-1-和为s的两个数字-简单-2022-01)
- [`剑指Offer No.057 2-和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-no057-2-和为s的连续正数序列-简单-2022-01)
- [`剑指Offer No.058 1-翻转单词顺序 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-no058-1-翻转单词顺序-简单-2022-01)

</details>

<details><summary><b> 技巧-双指针-快慢指针 [2] <a href="algorithms/topics/技巧-双指针-快慢指针.md">¶</a></b></summary>

- [`剑指Offer No.022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/技巧-双指针-快慢指针.md#剑指offer-no022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#剑指offer-no052-两个链表的第一个公共节点-简单-2022-01)

</details>

<details><summary><b> 技巧-双指针-滑动窗口 [1] <a href="algorithms/topics/技巧-双指针-滑动窗口.md">¶</a></b></summary>

- [`剑指Offer No.059 1-滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/技巧-双指针-滑动窗口.md#剑指offer-no059-1-滑动窗口的最大值-困难-2022-01)

</details>

<details><summary><b> 技巧-哈希表(Hash) [7] <a href="algorithms/topics/技巧-哈希表(Hash).md">¶</a></b></summary>

- [`LeetCode No.0001 两数之和 (简单, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-no0001-两数之和-简单-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`剑指Offer No.003 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no003-数组中重复的数字-简单-2021-11)
- [`剑指Offer No.035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.050 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no050-第一个只出现一次的字符-简单-2021-12)
- [`程序员面试金典 No.0102 判定是否互为字符重排 (简单, 2022-01)`](algorithms/topics/技巧-哈希表(Hash).md#程序员面试金典-no0102-判定是否互为字符重排-简单-2022-01)

</details>

<details><summary><b> 技巧-有限状态自动机 [1] <a href="algorithms/topics/技巧-有限状态自动机.md">¶</a></b></summary>

- [`剑指Offer No.020 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/技巧-有限状态自动机.md#剑指offer-no020-表示数值的字符串-中等-2021-11)

</details>

<details><summary><b> 技巧-贪心 [2] <a href="algorithms/topics/技巧-贪心.md">¶</a></b></summary>

- [`LeetCode No.0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/技巧-贪心.md#leetcode-no0300-最长递增子序列-中等-2022-01)
- [`剑指Offer No.014 1-剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/技巧-贪心.md#剑指offer-no014-1-剪绳子整数拆分-中等-2021-11)

</details>

<details><summary><b> 数据结构-二叉搜索树 [1] <a href="algorithms/topics/数据结构-二叉搜索树.md">¶</a></b></summary>

- [`剑指Offer No.068 1-二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/数据结构-二叉搜索树.md#剑指offer-no068-1-二叉搜索树的最近公共祖先-简单-2022-01)

</details>

<details><summary><b> 数据结构-二叉树 [18] <a href="algorithms/topics/数据结构-二叉树.md">¶</a></b></summary>

- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/数据结构-二叉树.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/数据结构-二叉树.md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/数据结构-二叉树.md#leetcode-no0437-路径总和3-中等-2021-10)
- [`剑指Offer No.007 重建二叉树 (中等, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no007-重建二叉树-中等-2021-11)
- [`剑指Offer No.026 树的子结构 (中等, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no026-树的子结构-中等-2021-11)
- [`剑指Offer No.027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no032-3-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer No.033 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no033-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer No.034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.037 序列化二叉树 (困难, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no037-序列化二叉树-困难-2021-12)
- [`剑指Offer No.054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no054-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer No.055 1-求二叉树的深度 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no055-1-求二叉树的深度-简单-2022-01)
- [`剑指Offer No.055 2-判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no055-2-判断是否为平衡二叉树-简单-2022-01)
- [`剑指Offer No.068 2-二叉树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-no068-2-二叉树的最近公共祖先-简单-2022-01)

</details>

<details><summary><b> 数据结构-堆、优先队列 [3] <a href="algorithms/topics/数据结构-堆、优先队列.md">¶</a></b></summary>

- [`剑指Offer No.040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-no040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-no041-数据流中的中位数-困难-2021-12)
- [`剑指Offer2 No.076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer2-no076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 数据结构-字符串 [8] <a href="algorithms/topics/数据结构-字符串.md">¶</a></b></summary>

- [`+模板 字符串 split切分 (简单, 模板库-基础)`](algorithms/topics/数据结构-字符串.md#模板-字符串-split切分-简单-模板库-基础)
- [`LeetCode No.0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/数据结构-字符串.md#leetcode-no0434-字符串中的单词数-简单-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#leetcode-no0859-亲密字符串-简单-2021-11)
- [`剑指Offer No.005 替换空格 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-no005-替换空格-简单-2021-11)
- [`剑指Offer No.019 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-no019-正则表达式匹配-困难-2021-11)
- [`剑指Offer No.020 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-no020-表示数值的字符串-中等-2021-11)
- [`剑指Offer No.058 2-左旋转字符串 (简单, 2022-01)`](algorithms/topics/数据结构-字符串.md#剑指offer-no058-2-左旋转字符串-简单-2022-01)
- [`剑指Offer No.067 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/数据结构-字符串.md#剑指offer-no067-把字符串转换成整数atoi-中等-2022-01)

</details>

<details><summary><b> 数据结构-数组、矩阵(二维数组) [4] <a href="algorithms/topics/数据结构-数组、矩阵(二维数组).md">¶</a></b></summary>

- [`剑指Offer No.021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no031-栈的压入弹出序列-中等-2021-11)

</details>

<details><summary><b> 数据结构-栈、队列 [9] <a href="algorithms/topics/数据结构-栈、队列.md">¶</a></b></summary>

- [`剑指Offer No.006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no032-3-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer No.059 2-队列的最大值 (中等, 2022-01)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no059-2-队列的最大值-中等-2022-01)

</details>

<details><summary><b> 数据结构-线段树、树状数组 [1] <a href="algorithms/topics/数据结构-线段树、树状数组.md">¶</a></b></summary>

- [`剑指Offer No.051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/数据结构-线段树、树状数组.md#剑指offer-no051-数组中的逆序对-困难-2022-01)

</details>

<details><summary><b> 数据结构-设计 [4] <a href="algorithms/topics/数据结构-设计.md">¶</a></b></summary>

- [`剑指Offer No.009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-no009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-no030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-设计.md#剑指offer-no041-数据流中的中位数-困难-2021-12)
- [`剑指Offer No.059 2-队列的最大值 (中等, 2022-01)`](algorithms/topics/数据结构-设计.md#剑指offer-no059-2-队列的最大值-中等-2022-01)

</details>

<details><summary><b> 数据结构-链表 [9] <a href="algorithms/topics/数据结构-链表.md">¶</a></b></summary>

- [`LeetCode No.0002 两数相加 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-no0002-两数相加-中等-2021-10)
- [`LeetCode No.0086 分隔链表 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-no0086-分隔链表-中等-2021-10)
- [`剑指Offer No.006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.018 删除链表的节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no018-删除链表的节点-简单-2021-11)
- [`剑指Offer No.022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.024 反转链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no024-反转链表-简单-2021-11)
- [`剑指Offer No.025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/数据结构-链表.md#剑指offer-no035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#剑指offer-no052-两个链表的第一个公共节点-简单-2022-01)

</details>

<details><summary><b> 算法-二分 [13] <a href="algorithms/topics/算法-二分.md">¶</a></b></summary>

- [`+模板 搜索 二分查找 (简单, 模板库-基础)`](algorithms/topics/算法-二分.md#模板-搜索-二分查找-简单-模板库-基础)
- [`LeetCode No.0029 两数相除 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0029-两数相除-中等-2021-10)
- [`LeetCode No.0033 搜索旋转排序数组 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0033-搜索旋转排序数组-中等-2021-10)
- [`LeetCode No.0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`剑指Offer No.004 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no004-二维数组中的查找-中等-2021-11)
- [`剑指Offer No.011 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no011-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer No.016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.053 1-求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-no053-1-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer No.053 2-在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-no053-2-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer2 No.001 整数除法 (中等, 2022-02)`](algorithms/topics/算法-二分.md#剑指offer2-no001-整数除法-中等-2022-02)
- [`剑指Offer2 No.069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/算法-二分.md#剑指offer2-no069-山峰数组的顶部-简单-2022-02)

</details>

<details><summary><b> 算法-分治 [4] <a href="algorithms/topics/算法-分治.md">¶</a></b></summary>

- [`剑指Offer No.007 重建二叉树 (中等, 2021-11)`](algorithms/topics/算法-分治.md#剑指offer-no007-重建二叉树-中等-2021-11)
- [`剑指Offer No.039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-分治.md#剑指offer-no039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/算法-分治.md#剑指offer-no051-数组中的逆序对-困难-2022-01)
- [`剑指Offer2 No.076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-分治.md#剑指offer2-no076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 算法-动态规划(记忆化搜索)、递推 [17] <a href="algorithms/topics/算法-动态规划(记忆化搜索)、递推.md">¶</a></b></summary>

- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0053 最大子数组和 (简单, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-no0053-最大子数组和-简单-2022-01)
- [`LeetCode No.0072 编辑距离 (困难, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-no0072-编辑距离-困难-2022-01)
- [`LeetCode No.0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-no0300-最长递增子序列-中等-2022-01)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-no0343-整数拆分-中等-2021-12)
- [`剑指Offer No.010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.010 斐波那契数列-2（青蛙跳台阶） (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no010-斐波那契数列-2青蛙跳台阶-简单-2021-11)
- [`剑指Offer No.014 1-剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no014-1-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer No.019 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no019-正则表达式匹配-困难-2021-11)
- [`剑指Offer No.042 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no042-连续子数组的最大和-简单-2021-12)
- [`剑指Offer No.046 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no046-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer No.047 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no047-礼物的最大价值-中等-2021-12)
- [`剑指Offer No.048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.049 丑数 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no049-丑数-中等-2021-12)
- [`剑指Offer No.060 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no060-n个骰子的点数-中等-2022-01)
- [`剑指Offer No.062 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-no062-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)

</details>

<details><summary><b> 算法-广度优先搜索(BFS) [3] <a href="algorithms/topics/算法-广度优先搜索(BFS).md">¶</a></b></summary>

- [`剑指Offer No.032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no032-3-层序遍历二叉树之字形遍历-简单-2021-11)

</details>

<details><summary><b> 算法-排序 [6] <a href="algorithms/topics/算法-排序.md">¶</a></b></summary>

- [`剑指Offer No.039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.045 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no045-把数组排成最小的数-中等-2021-12)
- [`剑指Offer No.061 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/算法-排序.md#剑指offer-no061-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer2 No.076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-排序.md#剑指offer2-no076-数组中的第k大的数字-中等-2022-02)
- [`程序员面试金典 No.0101 判定字符是否唯一 (简单, 2022-01)`](algorithms/topics/算法-排序.md#程序员面试金典-no0101-判定字符是否唯一-简单-2022-01)

</details>

<details><summary><b> 算法-深度优先搜索(DFS) [10] <a href="algorithms/topics/算法-深度优先搜索(DFS).md">¶</a></b></summary>

- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-no0437-路径总和3-中等-2021-10)
- [`剑指Offer No.006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.013 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no013-机器人的运动范围-中等-2021-11)
- [`剑指Offer No.017 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no017-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer No.034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no054-二叉搜索树的第k大节点-简单-2022-01)

</details>

<details><summary><b> 算法-递归、迭代 [14] <a href="algorithms/topics/算法-递归、迭代.md">¶</a></b></summary>

- [`LeetCode No.0021 合并两个有序链表 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-no0021-合并两个有序链表-简单-2021-10)
- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`剑指Offer No.006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.019 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no019-正则表达式匹配-困难-2021-11)
- [`剑指Offer No.024 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no024-反转链表-简单-2021-11)
- [`剑指Offer No.024 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no024-反转链表-简单-2021-11)
- [`剑指Offer No.025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.026 树的子结构 (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no026-树的子结构-中等-2021-11)
- [`剑指Offer No.027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.064 求1~n的和 (中等, 2022-01)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no064-求1n的和-中等-2022-01)

</details>


</details>

---

Coding Lab
---

### Work Utils [¶](src/README.md#work-utils)

- [`find_best_threshold: 搜索最佳阈值（二分类）`](src/README.md#find_best_threshold-搜索最佳阈值二分类)
- [`BertTokenizer: Bert 分词器`](src/README.md#berttokenizer-bert-分词器)
- [`ner_result_parse: NER 结果解析（基于 BIO 格式）`](src/README.md#ner_result_parse-ner-结果解析基于-bio-格式)
- [`split: 将数据按比例切分`](src/README.md#split-将数据按比例切分)
- [`ImageCheck: 图片完整性检查`](src/README.md#imagecheck-图片完整性检查)
- [`get_real_ext: 获取图像文件的真实后缀`](src/README.md#get_real_ext-获取图像文件的真实后缀)


### Pytorch Models [¶](src/README.md#pytorch-models)

- [`DualNet: 双塔结构`](src/README.md#dualnet-双塔结构)
- [`SiameseNet: 孪生网络，基于双塔结构`](src/README.md#siamesenet-孪生网络基于双塔结构)
- [`SimCSE: SimCSE`](src/README.md#simcse-simcse)
- [`Bert: Bert by Pytorch`](src/README.md#bert-bert-by-pytorch)
- [`BertPretrain: Bert 预训练（MLM + NSP）`](src/README.md#bertpretrain-bert-预训练mlm-nsp)
- [`SentenceBert: Bert 句向量`](src/README.md#sentencebert-bert-句向量)
- [`BertSequenceTagging: Bert 序列标注`](src/README.md#bertsequencetagging-bert-序列标注)
- [`BertTextClassification: Bert 文本分类`](src/README.md#berttextclassification-bert-文本分类)
- [`LayerNorm: Layer Normalization`](src/README.md#layernorm-layer-normalization)


### Pytorch Utils [¶](src/README.md#pytorch-utils)

- [`DictTensorDataset: 字典格式的 Dataset`](src/README.md#dicttensordataset-字典格式的-dataset)
- [`ToyDataLoader: 简化创建 DataLoader 的过程`](src/README.md#toydataloader-简化创建-dataloader-的过程)
- [`BertDataLoader: 简化 Bert 训练数据的加载`](src/README.md#bertdataloader-简化-bert-训练数据的加载)
- [`ContrastiveLoss: 对比损失（默认距离函数为欧几里得距离）`](src/README.md#contrastiveloss-对比损失默认距离函数为欧几里得距离)
- [`CrossEntropyLoss: 交叉熵`](src/README.md#crossentropyloss-交叉熵)
- [`TripletLoss: Triplet 损失，常用于无监督学习、few-shot 学习`](src/README.md#tripletloss-triplet-损失常用于无监督学习few-shot-学习)
- [`FGM: Fast Gradient Method (对抗训练)`](src/README.md#fgm-fast-gradient-method-对抗训练)
- [`PGM: Projected Gradient Method (对抗训练)`](src/README.md#pgm-projected-gradient-method-对抗训练)
- [`Trainer: Trainer 基类`](src/README.md#trainer-trainer-基类)
- [`set_seed: 设置全局随机数种子，使实验可复现`](src/README.md#set_seed-设置全局随机数种子使实验可复现)
- [`init_weights: 默认参数初始化`](src/README.md#init_weights-默认参数初始化)


### Python Utils [¶](src/README.md#python-utils)

- [`simple_argparse: 一个简化版 argparse`](src/README.md#simple_argparse-一个简化版-argparse)
- [`ArrayDict: 数组字典，支持 slice`](src/README.md#arraydict-数组字典支持-slice)
- [`ValueArrayDict: 数组字典，支持 slice，且操作 values`](src/README.md#valuearraydict-数组字典支持-slice且操作-values)
- [`BunchDict: 基于 dict 实现 Bunch 模式`](src/README.md#bunchdict-基于-dict-实现-bunch-模式)
- [`FieldBunchDict: 基于 dataclass 的 BunchDict`](src/README.md#fieldbunchdict-基于-dataclass-的-bunchdict)
- [`ls_dir_recur: 递归遍历目录下的所有文件`](src/README.md#ls_dir_recur-递归遍历目录下的所有文件)
- [`files_concat: 文件拼接`](src/README.md#files_concat-文件拼接)
- [`get_caller_name: 获取调用者的名称`](src/README.md#get_caller_name-获取调用者的名称)
- [`function_test_dn: 函数测试装饰器`](src/README.md#function_test_dn-函数测试装饰器)


---

