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

- [ ] 【`2021.11.11`】pytorch_trainer: 为 EvaluateCallback 添加各种预定义评估指标，如 acc、f1 等，目前只有 loss；
- [ ] 【`2021.11.11`】论文：What does BERT learn about the structure of language? —— Bert 各层的含义；
- [ ] 【`2021.11.10`】bert-tokenizer 自动识别 `[MASK]` 等特殊标识；
- [ ] 【`2021.11.07`】面试笔记：通识问题/项目问题
- [ ] 【`2021.10.22`】max_batch_size 估算

<details><summary><b> Done </b></summary>

- [x] 【`2021.11.06-2021.11.12`】优化 auto-readme，使用上一次的 commit info，而不是默认 'Auto-README'
    - 参考：`git commit -m "$(git log -"$(git rev-list origin/master..master --count)" --pretty=%B | cat)"`
    - 说明：使用 origin/master 到 master 之间所有的 commit 信息作为这次的 message；
- [x] 【`2021.11.10-2021.11.11`】bert 支持加载指定层 -> `_test_load_appointed_layers()`
- [x] 【`2021.11.04-2021.11.08`】把 __test.py 文件自动加入文档测试（放弃）
    - 有些测试比较耗时，不需要全部加入自动测试；
    - __test.py 针对的是存在相对引用的模块，如果这些模块有改动，会即时测试，所以也不需要自动测试
- [x] 【`2021.11.01-2021.11.03`】[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 代码阅读

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

<details><summary><b> 题集-LeetCode [24] <a href="algorithms/topics/题集-LeetCode.md">¶</a></b></summary>

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
- [`LeetCode No.0086 分隔链表 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0086-分隔链表-中等-2021-10)
- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`LeetCode No.0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/题集-LeetCode.md#leetcode-no0343-整数拆分-中等-2021-12)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0434-字符串中的单词数-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0437-路径总和3-中等-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`LeetCode No.0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/题集-LeetCode.md#leetcode-no0496-下一个更大元素-简单-2021-11)
- [`LeetCode No.0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/题集-LeetCode.md#leetcode-no0611-有效三角形的个数-中等-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/题集-LeetCode.md#leetcode-no0859-亲密字符串-简单-2021-11)

</details>

<details><summary><b> 题集-剑指Offer [58] <a href="algorithms/topics/题集-剑指Offer.md">¶</a></b></summary>

- [`剑指Offer No.0003 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0003-数组中重复的数字-简单-2021-11)
- [`剑指Offer No.0004 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0004-二维数组中的查找-中等-2021-11)
- [`剑指Offer No.0005 替换空格 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0005-替换空格-简单-2021-11)
- [`剑指Offer No.0006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.0007 重建二叉树 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0007-重建二叉树-中等-2021-11)
- [`剑指Offer No.0009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.0010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.0010 斐波那契数列-2（青蛙跳台阶） (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0010-斐波那契数列-2青蛙跳台阶-简单-2021-11)
- [`剑指Offer No.0011 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0011-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer No.0012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.0013 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0013-机器人的运动范围-中等-2021-11)
- [`剑指Offer No.0014 剪绳子1（整数拆分） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0014-剪绳子1整数拆分-中等-2021-11)
- [`剑指Offer No.0015 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0015-二进制中1的个数-简单-2021-11)
- [`剑指Offer No.0016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.0017 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0017-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer No.0018 删除链表的节点 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0018-删除链表的节点-简单-2021-11)
- [`剑指Offer No.0021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.0022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.0024 反转链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0024-反转链表-简单-2021-11)
- [`剑指Offer No.0025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.0026 树的子结构 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0026-树的子结构-中等-2021-11)
- [`剑指Offer No.0027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.0028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.0029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.0030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.0031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.0032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0032-3-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer No.0033 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0033-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer No.0034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.0035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.0036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.0037 序列化二叉树 (困难, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0037-序列化二叉树-困难-2021-12)
- [`剑指Offer No.0038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.0039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.0040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.0041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0041-数据流中的中位数-困难-2021-12)
- [`剑指Offer No.0042 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0042-连续子数组的最大和-简单-2021-12)
- [`剑指Offer No.0044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.0045 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0045-把数组排成最小的数-中等-2021-12)
- [`剑指Offer No.0046 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0046-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer No.0047 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0047-礼物的最大价值-中等-2021-12)
- [`剑指Offer No.0048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.0049 丑数 (中等, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0049-丑数-中等-2021-12)
- [`剑指Offer No.0050 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0050-第一个只出现一次的字符-简单-2021-12)
- [`剑指Offer No.0051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0051-数组中的逆序对-困难-2022-01)
- [`剑指Offer No.0052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0052-两个链表的第一个公共节点-简单-2022-01)
- [`剑指Offer No.0053 1-求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0053-1-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer No.0053 2-在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0053-2-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer No.0054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0054-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer No.0055 1-求二叉树的深度 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0055-1-求二叉树的深度-简单-2022-01)
- [`剑指Offer No.0055 2-判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0055-2-判断是否为平衡二叉树-简单-2022-01)
- [`剑指Offer No.0056 1-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0056-1-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer No.0057 1-和为s的两个数字 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0057-1-和为s的两个数字-简单-2022-01)
- [`剑指Offer No.0057 2-和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0057-2-和为s的连续正数序列-简单-2022-01)
- [`剑指Offer No.0063 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0063-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer No.0067 把字符串转换成整数 (中等, 2022-01)`](algorithms/topics/题集-剑指Offer.md#剑指offer-no0067-把字符串转换成整数-中等-2022-01)

</details>

<details><summary><b> 题集-剑指Offer(突击版) [2] <a href="algorithms/topics/题集-剑指Offer(突击版).md">¶</a></b></summary>

- [`剑指Offer(突击版) No.0069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/题集-剑指Offer(突击版).md#剑指offer突击版-no0069-山峰数组的顶部-简单-2022-02)
- [`剑指Offer(突击版) No.0076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/题集-剑指Offer(突击版).md#剑指offer突击版-no0076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 题集-经典问题&代码 [12] <a href="algorithms/topics/题集-经典问题&代码.md">¶</a></b></summary>

- [`剑指Offer No.0007 重建二叉树 (中等, 2021-11)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0007-重建二叉树-中等-2021-11)
- [`剑指Offer No.0016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.0024 反转链表 (简单, 2021-11)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0024-反转链表-简单-2021-11)
- [`剑指Offer No.0029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.0031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.0035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.0036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.0038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.0039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.0040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.0049 丑数 (中等, 2021-12)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0049-丑数-中等-2021-12)
- [`剑指Offer No.0051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/题集-经典问题&代码.md#剑指offer-no0051-数组中的逆序对-困难-2022-01)

</details>

<details><summary><b>More ...<a href="algorithms/README.md">¶</a></b></summary>

<details><summary><b> 基础-模拟、数学、找规律 [12] <a href="algorithms/topics/基础-模拟、数学、找规律.md">¶</a></b></summary>

- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0343-整数拆分-中等-2021-12)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#leetcode-no0859-亲密字符串-简单-2021-11)
- [`剑指Offer No.0014 剪绳子1（整数拆分） (中等, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0014-剪绳子1整数拆分-中等-2021-11)
- [`剑指Offer No.0029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.0039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.0044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.0044 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0044-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer No.0063 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0063-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer No.0067 把字符串转换成整数 (中等, 2022-01)`](algorithms/topics/基础-模拟、数学、找规律.md#剑指offer-no0067-把字符串转换成整数-中等-2022-01)

</details>

<details><summary><b> 技巧-位运算 [4] <a href="algorithms/topics/技巧-位运算.md">¶</a></b></summary>

- [`LeetCode No.0029 两数相除 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-no0029-两数相除-中等-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`剑指Offer No.0015 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/技巧-位运算.md#剑指offer-no0015-二进制中1的个数-简单-2021-11)
- [`剑指Offer No.0056 1-数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-no0056-1-数组中数字出现的次数-中等-2022-01)

</details>

<details><summary><b> 技巧-前缀和 [1] <a href="algorithms/topics/技巧-前缀和.md">¶</a></b></summary>

- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/技巧-前缀和.md#leetcode-no0437-路径总和3-中等-2021-10)

</details>

<details><summary><b> 技巧-单调栈 [1] <a href="algorithms/topics/技巧-单调栈.md">¶</a></b></summary>

- [`LeetCode No.0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/技巧-单调栈.md#leetcode-no0496-下一个更大元素-简单-2021-11)

</details>

<details><summary><b> 技巧-双指针、滑动窗口 [12] <a href="algorithms/topics/技巧-双指针、滑动窗口.md">¶</a></b></summary>

- [`LeetCode No.0011 盛最多水的容器 (中等, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0011-盛最多水的容器-中等-2021-10)
- [`LeetCode No.0015 三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0015-三数之和-中等-2021-10)
- [`LeetCode No.0016 最接近的三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0016-最接近的三数之和-中等-2021-10)
- [`LeetCode No.0042 接雨水 (困难, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0042-接雨水-困难-2021-10)
- [`LeetCode No.0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode No.0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/技巧-双指针、滑动窗口.md#leetcode-no0611-有效三角形的个数-中等-2021-10)
- [`剑指Offer No.0021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.0022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.0048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.0052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0052-两个链表的第一个公共节点-简单-2022-01)
- [`剑指Offer No.0057 1-和为s的两个数字 (简单, 2022-01)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0057-1-和为s的两个数字-简单-2022-01)
- [`剑指Offer No.0057 2-和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/技巧-双指针、滑动窗口.md#剑指offer-no0057-2-和为s的连续正数序列-简单-2022-01)

</details>

<details><summary><b> 技巧-哈希表(Hash) [6] <a href="algorithms/topics/技巧-哈希表(Hash).md">¶</a></b></summary>

- [`LeetCode No.0001 两数之和 (简单, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-no0001-两数之和-简单-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`剑指Offer No.0003 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no0003-数组中重复的数字-简单-2021-11)
- [`剑指Offer No.0035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no0035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.0048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no0048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.0050 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-no0050-第一个只出现一次的字符-简单-2021-12)

</details>

<details><summary><b> 技巧-贪心 [1] <a href="algorithms/topics/技巧-贪心.md">¶</a></b></summary>

- [`剑指Offer No.0014 剪绳子1（整数拆分） (中等, 2021-11)`](algorithms/topics/技巧-贪心.md#剑指offer-no0014-剪绳子1整数拆分-中等-2021-11)

</details>

<details><summary><b> 数据结构-堆、优先队列 [3] <a href="algorithms/topics/数据结构-堆、优先队列.md">¶</a></b></summary>

- [`剑指Offer No.0040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-no0040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.0041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-no0041-数据流中的中位数-困难-2021-12)
- [`剑指Offer(突击版) No.0076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer突击版-no0076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 数据结构-字符串 [4] <a href="algorithms/topics/数据结构-字符串.md">¶</a></b></summary>

- [`LeetCode No.0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/数据结构-字符串.md#leetcode-no0434-字符串中的单词数-简单-2021-10)
- [`LeetCode No.0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#leetcode-no0859-亲密字符串-简单-2021-11)
- [`剑指Offer No.0005 替换空格 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-no0005-替换空格-简单-2021-11)
- [`剑指Offer No.0067 把字符串转换成整数 (中等, 2022-01)`](algorithms/topics/数据结构-字符串.md#剑指offer-no0067-把字符串转换成整数-中等-2022-01)

</details>

<details><summary><b> 数据结构-数组、矩阵(二维数组) [4] <a href="algorithms/topics/数据结构-数组、矩阵(二维数组).md">¶</a></b></summary>

- [`剑指Offer No.0021 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no0021-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer No.0029 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no0029-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer No.0030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no0030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.0031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-no0031-栈的压入弹出序列-中等-2021-11)

</details>

<details><summary><b> 数据结构-栈、队列 [8] <a href="algorithms/topics/数据结构-栈、队列.md">¶</a></b></summary>

- [`剑指Offer No.0006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.0009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.0009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.0030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.0031 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0031-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer No.0032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-no0032-3-层序遍历二叉树之字形遍历-简单-2021-11)

</details>

<details><summary><b> 数据结构-树、二叉树 [17] <a href="algorithms/topics/数据结构-树、二叉树.md">¶</a></b></summary>

- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/数据结构-树、二叉树.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/数据结构-树、二叉树.md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/数据结构-树、二叉树.md#leetcode-no0437-路径总和3-中等-2021-10)
- [`剑指Offer No.0007 重建二叉树 (中等, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0007-重建二叉树-中等-2021-11)
- [`剑指Offer No.0026 树的子结构 (中等, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0026-树的子结构-中等-2021-11)
- [`剑指Offer No.0027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.0028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.0032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0032-3-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer No.0033 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0033-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer No.0034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.0036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0036-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer No.0037 序列化二叉树 (困难, 2021-12)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0037-序列化二叉树-困难-2021-12)
- [`剑指Offer No.0054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0054-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer No.0055 1-求二叉树的深度 (简单, 2022-01)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0055-1-求二叉树的深度-简单-2022-01)
- [`剑指Offer No.0055 2-判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/数据结构-树、二叉树.md#剑指offer-no0055-2-判断是否为平衡二叉树-简单-2022-01)

</details>

<details><summary><b> 数据结构-线段树、树状数组 [1] <a href="algorithms/topics/数据结构-线段树、树状数组.md">¶</a></b></summary>

- [`剑指Offer No.0051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/数据结构-线段树、树状数组.md#剑指offer-no0051-数组中的逆序对-困难-2022-01)

</details>

<details><summary><b> 数据结构-设计 [3] <a href="algorithms/topics/数据结构-设计.md">¶</a></b></summary>

- [`剑指Offer No.0009 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-no0009-用两个栈实现队列-简单-2021-11)
- [`剑指Offer No.0030 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-no0030-包含min函数的栈-简单-2021-11)
- [`剑指Offer No.0041 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-设计.md#剑指offer-no0041-数据流中的中位数-困难-2021-12)

</details>

<details><summary><b> 数据结构-链表 [9] <a href="algorithms/topics/数据结构-链表.md">¶</a></b></summary>

- [`LeetCode No.0002 两数相加 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-no0002-两数相加-中等-2021-10)
- [`LeetCode No.0086 分隔链表 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-no0086-分隔链表-中等-2021-10)
- [`剑指Offer No.0006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no0006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.0018 删除链表的节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no0018-删除链表的节点-简单-2021-11)
- [`剑指Offer No.0022 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no0022-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer No.0024 反转链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no0024-反转链表-简单-2021-11)
- [`剑指Offer No.0025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-no0025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.0035 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/数据结构-链表.md#剑指offer-no0035-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.0052 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#剑指offer-no0052-两个链表的第一个公共节点-简单-2022-01)

</details>

<details><summary><b> 算法-二分 [11] <a href="algorithms/topics/算法-二分.md">¶</a></b></summary>

- [`LeetCode No.0029 两数相除 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0029-两数相除-中等-2021-10)
- [`LeetCode No.0033 搜索旋转排序数组 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0033-搜索旋转排序数组-中等-2021-10)
- [`LeetCode No.0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode No.0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode No.0441 排列硬币 (简单, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-no0441-排列硬币-简单-2021-10)
- [`剑指Offer No.0004 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no0004-二维数组中的查找-中等-2021-11)
- [`剑指Offer No.0011 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no0011-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer No.0016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-no0016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.0053 1-求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-no0053-1-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer No.0053 2-在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-no0053-2-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer(突击版) No.0069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/算法-二分.md#剑指offer突击版-no0069-山峰数组的顶部-简单-2022-02)

</details>

<details><summary><b> 算法-分治 [4] <a href="algorithms/topics/算法-分治.md">¶</a></b></summary>

- [`剑指Offer No.0007 重建二叉树 (中等, 2021-11)`](algorithms/topics/算法-分治.md#剑指offer-no0007-重建二叉树-中等-2021-11)
- [`剑指Offer No.0039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-分治.md#剑指offer-no0039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.0051 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/算法-分治.md#剑指offer-no0051-数组中的逆序对-困难-2022-01)
- [`剑指Offer(突击版) No.0076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-分治.md#剑指offer突击版-no0076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 算法-动态规划(DP、记忆化搜索) [11] <a href="algorithms/topics/算法-动态规划(DP、记忆化搜索).md">¶</a></b></summary>

- [`LeetCode No.0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#leetcode-no0005-最长回文子串-中等-2021-10)
- [`LeetCode No.0343 整数拆分 (中等, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#leetcode-no0343-整数拆分-中等-2021-12)
- [`剑指Offer No.0010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.0010 斐波那契数列-1 (简单, 2021-11)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0010-斐波那契数列-1-简单-2021-11)
- [`剑指Offer No.0010 斐波那契数列-2（青蛙跳台阶） (简单, 2021-11)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0010-斐波那契数列-2青蛙跳台阶-简单-2021-11)
- [`剑指Offer No.0014 剪绳子1（整数拆分） (中等, 2021-11)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0014-剪绳子1整数拆分-中等-2021-11)
- [`剑指Offer No.0042 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0042-连续子数组的最大和-简单-2021-12)
- [`剑指Offer No.0046 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0046-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer No.0047 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0047-礼物的最大价值-中等-2021-12)
- [`剑指Offer No.0048 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0048-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.0049 丑数 (中等, 2021-12)`](algorithms/topics/算法-动态规划(DP、记忆化搜索).md#剑指offer-no0049-丑数-中等-2021-12)

</details>

<details><summary><b> 算法-广度优先搜索(BFS) [3] <a href="algorithms/topics/算法-广度优先搜索(BFS).md">¶</a></b></summary>

- [`剑指Offer No.0032 1-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no0032-1-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 2-层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no0032-2-层序遍历二叉树-简单-2021-11)
- [`剑指Offer No.0032 3-层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-no0032-3-层序遍历二叉树之字形遍历-简单-2021-11)

</details>

<details><summary><b> 算法-排序 [4] <a href="algorithms/topics/算法-排序.md">¶</a></b></summary>

- [`剑指Offer No.0039 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no0039-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer No.0040 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no0040-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer No.0045 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-no0045-把数组排成最小的数-中等-2021-12)
- [`剑指Offer(突击版) No.0076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-排序.md#剑指offer突击版-no0076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 算法-深度优先搜索(DFS) [10] <a href="algorithms/topics/算法-深度优先搜索(DFS).md">¶</a></b></summary>

- [`LeetCode No.0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-no0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode No.0437 路径总和3 (中等, 2021-10)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-no0437-路径总和3-中等-2021-10)
- [`剑指Offer No.0006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.0012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.0012 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0012-矩阵中的路径-中等-2021-11)
- [`剑指Offer No.0013 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0013-机器人的运动范围-中等-2021-11)
- [`剑指Offer No.0017 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0017-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer No.0034 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0034-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer No.0038 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0038-字符串的排列全排列-中等-2021-12)
- [`剑指Offer No.0054 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-no0054-二叉搜索树的第k大节点-简单-2022-01)

</details>

<details><summary><b> 算法-递归、迭代 [12] <a href="algorithms/topics/算法-递归、迭代.md">¶</a></b></summary>

- [`LeetCode No.0021 合并两个有序链表 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-no0021-合并两个有序链表-简单-2021-10)
- [`LeetCode No.0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-no0104-二叉树的最大深度-简单-2021-10)
- [`剑指Offer No.0006 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0006-从尾到头打印链表-简单-2021-11)
- [`剑指Offer No.0016 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0016-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer No.0024 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0024-反转链表-简单-2021-11)
- [`剑指Offer No.0024 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0024-反转链表-简单-2021-11)
- [`剑指Offer No.0025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.0025 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0025-合并两个排序的链表-简单-2021-11)
- [`剑指Offer No.0026 树的子结构 (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0026-树的子结构-中等-2021-11)
- [`剑指Offer No.0027 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0027-二叉树的镜像-简单-2021-11)
- [`剑指Offer No.0028 对称的二叉树 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0028-对称的二叉树-简单-2021-11)
- [`剑指Offer No.0036 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/算法-递归、迭代.md#剑指offer-no0036-二叉搜索树与双向链表-中等-2021-12)

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

