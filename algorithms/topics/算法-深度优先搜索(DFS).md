# 深度优先搜索(DFS)

<!-- Tag: 深度优先搜索(递归)、深度优先搜索、DFS、DFS+回溯、DFS+剪枝 -->

## 普通的 DFS
<!-- TODO -->

## 回溯的 DFS
<!-- TODO -->

Problems
---
- [`LeetCode 0111 二叉树的最小深度 (简单, 2021-10)`](#leetcode-0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](#leetcode-0437-路径总和iii-中等-2022-02)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`](#剑指offer-1200-矩阵中的路径-中等-2021-11)
- [`剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`](#剑指offer-1200-矩阵中的路径-中等-2021-11)
- [`剑指Offer 1300 机器人的运动范围 (中等, 2021-11)`](#剑指offer-1300-机器人的运动范围-中等-2021-11)
- [`剑指Offer 1700 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](#剑指offer-1700-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer 3400 二叉树中和为某一值的路径 (中等, 2021-12)`](#剑指offer-3400-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer 3800 字符串的排列（全排列） (中等, 2021-12)`](#剑指offer-3800-字符串的排列全排列-中等-2021-12)
- [`剑指Offer 5400 二叉搜索树的第k大节点 (简单, 2022-01)`](#剑指offer-5400-二叉搜索树的第k大节点-简单-2022-01)
- [`牛客 0005 二叉树根节点到叶子节点的所有路径和 (中等, 2022-01)`](#牛客-0005-二叉树根节点到叶子节点的所有路径和-中等-2022-01)
- [`牛客 0008 二叉树中和为某一值的路径(二) (中等, 2022-01)`](#牛客-0008-二叉树中和为某一值的路径二-中等-2022-01)
- [`牛客 0009 二叉树中和为某一值的路径(一) (简单, 2022-01)`](#牛客-0009-二叉树中和为某一值的路径一-简单-2022-01)
- [`牛客 0020 数字字符串转化成IP地址 (中等, 2022-01)`](#牛客-0020-数字字符串转化成ip地址-中等-2022-01)
- [`牛客 0045 实现二叉树先序、中序、后序遍历 (中等, 2022-03)`](#牛客-0045-实现二叉树先序中序后序遍历-中等-2022-03)

---

### `LeetCode 0111 二叉树的最小深度 (简单, 2021-10)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](合集-LeetCode.md)

<!--{
    "tags": ["二叉树", "DFS"],
    "来源": "LeetCode",
    "编号": "0111",
    "难度": "简单",
    "标题": "二叉树的最小深度"
}-->

<summary><b>问题描述</b></summary>

```txt
给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

示例：
    给定二叉树 [3,9,20,null,null,15,7]，
        3
       / \
      9  20
        /  \
       15   7
    返回它的最小深度 2 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/minimum-depth-of-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<summary><b>思路</b></summary>

- 深度优先搜索，记录过程中的最小深度；

<details><summary><b>Python：深度优先搜索</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        """"""
        if not root:  # 尾递归1
            return 0

        if not root.left and not root.right:  # 尾递归 2 *
            return 1
        
        min_depth = 10**5 + 10
        if root.left:
            min_depth = min(self.minDepth(root.left), min_depth)
        if root.right:
            min_depth = min(self.minDepth(root.right), min_depth)
        
        return min_depth + 1
```

</details>

---

### `LeetCode 0437 路径总和III (中等, 2022-02)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![深度优先搜索](https://img.shields.io/badge/深度优先搜索-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![前缀和](https://img.shields.io/badge/前缀和-lightgray.svg)](技巧-前缀和.md)
[![TreeDP](https://img.shields.io/badge/TreeDP-lightgray.svg)](技巧-自底向上的递归技巧.md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](合集-LeetCode.md)

<!--{
    "tags": ["二叉树", "深度优先搜索", "前缀和", "TreeDP"],
    "来源": "LeetCode",
    "编号": "0437",
    "难度": "中等",
    "标题": "路径总和III",
    "公司": []
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
```
> [437. 路径总和 III - 力扣（LeetCode）](https://leetcode-cn.com/problems/path-sum-iii/)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：先序遍历</b></summary>

- 先序遍历每个节点，每个节点再先序遍历找目标值；

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:  # noqa
        """"""
        if root is None:
            return 0

        def dfs(x, rest):
            if not x:
                return 0

            ans = 0 if x.val != rest else 1  # 如果相等说明，从头结点开始到该节点可以形成一条路径

            # 继续遍历左右子树
            rest -= x.val
            ans += dfs(x.left, rest)
            ans += dfs(x.right, rest)
            rest += x.val  # 回溯
            return ans

        # dfs 是一个先序遍历
        ret = dfs(root, targetSum)
        # pathSum 本身也是一个先序遍历，相当于对每个点都做一次 dfs
        ret += self.pathSum(root.left, targetSum)
        ret += self.pathSum(root.right, targetSum)

        return ret
```

</details>


<summary><b>思路2：先序遍历+前缀和（最优）</b></summary>

> [【宫水三叶】一题双解 :「DFS」&「前缀和」 - 路径总和 III - 力扣（LeetCode）](https://leetcode-cn.com/problems/path-sum-iii/solution/gong-shui-san-xie-yi-ti-shuang-jie-dfs-q-usa7/)

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        from collections import defaultdict
        self.prefix = defaultdict(int)  # 保存前缀和
        self.prefix[0] = 1
        self.targetSum = targetSum

        def dfs(x, preSum):
            if not x: return 0

            ret = 0
            preSum += x.val
            ret += self.prefix[preSum - targetSum]

            self.prefix[preSum] += 1
            ret += dfs(x.left, preSum)
            ret += dfs(x.right, preSum)
            self.prefix[preSum] -= 1

            return ret

        return dfs(root, 0)
```

</details>


<summary><b>思路3：后序遍历（树形DP）（推荐）</b></summary>

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        from collections import defaultdict
        self.prefix = defaultdict(int)  # 保存前缀和
        self.prefix[0] = 1
        self.targetSum = targetSum

        def dfs(x, preSum):
            if not x: return 0

            ret = 0
            preSum += x.val
            ret += self.prefix[preSum - targetSum]

            self.prefix[preSum] += 1
            ret += dfs(x.left, preSum)
            ret += dfs(x.right, preSum)
            self.prefix[preSum] -= 1

            return ret

        return dfs(root, 0)
```

</details>

---

### `剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`

[![链表](https://img.shields.io/badge/链表-lightgray.svg)](数据结构-链表.md)
[![栈](https://img.shields.io/badge/栈-lightgray.svg)](数据结构-栈、队列.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![递归](https://img.shields.io/badge/递归-lightgray.svg)](算法-递归、迭代.md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["链表", "栈", "DFS", "递归"],
    "来源": "剑指Offer",
    "编号": "0600",
    "难度": "简单",
    "标题": "从尾到头打印链表"
}-->

<summary><b>问题简述</b></summary>

```txt
从尾到头打印链表（用数组返回）
```

<details><summary><b>详细描述</b></summary>

```txt
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

示例 1：
    输入：head = [1,3,2]
    输出：[2,3,1]

限制：
    0 <= 链表长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

</details>


<summary><b>思路</b></summary>

- 法1）利用栈，顺序入栈，然后依次出栈即可
- 法2）利用深度优先遍历思想（二叉树的先序遍历）


<details><summary><b>Python：栈</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        
        # ret = []
        # for _ in range(len(stack)):  # 相当于逆序遍历
        #     ret.append(stack.pop())
        # return ret
        return stack[::-1]  # 与以上代码等价
```

</details>

<details><summary><b>Python：DFS、递归</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if head is None:
            return []

        ret = self.reversePrint(head.next)
        ret.append(head.val)

        return ret
```

</details>

---

### `剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![DFS+回溯](https://img.shields.io/badge/DFS+回溯-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["DFS", "DFS+回溯"],
    "来源": "剑指Offer",
    "编号": "1200",
    "难度": "中等",
    "标题": "矩阵中的路径"
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个 m x n 二维字符矩阵 board 和字符串 word。如果 word 存在于网格中，返回 true ；否则，返回 false 。

其中单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

示例 1：
    输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
    输出：true
示例 2：
    输入：board = [["a","b"],["c","d"]], word = "abcd"
    输出：false
 
提示：
    1 <= board.length <= 200
    1 <= board[i].length <= 200
    board 和 word 仅由大小写英文字母组成
 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<div align="center"><img src="../_assets/剑指Offer_0012_中等_矩阵中的路径-示例.jpeg" height="200" /></div>

</details>

<summary><b>思路</b></summary>

- 棋盘搜索，非常典型的 DFS + 回溯问题；

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<details><summary><b>Python：DFS + 回溯</b></summary>

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if len(board) < 1:
            return False

        m, n = len(board), len(board[0])

        # 使用内部函数，可以减少一些参数的传递，同时比成员方法更简洁
        def dfs(i, j, k):  # i, j, k 分别表示 board[i][j] 和 word[k]
            if not 0 <= i < m or not 0 <= j < n:  # 先判断是否越界
                return False

            if board[i][j] != word[k]:  # 这一步可以合并到越界判断，但会损失一些可读性，故分离出来单独判断
                return False
            else:  # board[i][j] == word[k]:  # 如果当前位置字符相同，继续深度搜索
                if k == len(word) - 1:  # 如果字符已经全部匹配成功，返回 True
                    return True

                # 置空，表示该位置已访问过；一些代码中会使用一个新的矩阵记录位置是否访问，这里直接在原矩阵上标记
                board[i][j] = ''
                # 继续遍历 4 个方向
                flag = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
                # 这一步是容易忽略的：因为需要回溯，所以必须还原该位置的元素
                board[i][j] = word[k]

                return flag

        # board 中每一个位置都可能是起始位置，所以要循环遍历
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
```

</details>

---

### `剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![DFS+回溯](https://img.shields.io/badge/DFS+回溯-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["DFS", "DFS+回溯"],
    "来源": "剑指Offer",
    "编号": "1200",
    "难度": "中等",
    "标题": "矩阵中的路径"
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个 m x n 二维字符矩阵 board 和字符串 word。如果 word 存在于网格中，返回 true ；否则，返回 false 。

其中单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

示例 1：
    输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
    输出：true
示例 2：
    输入：board = [["a","b"],["c","d"]], word = "abcd"
    输出：false
 
提示：
    1 <= board.length <= 200
    1 <= board[i].length <= 200
    board 和 word 仅由大小写英文字母组成
 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<div align="center"><img src="../_assets/剑指Offer_0012_中等_矩阵中的路径-示例.jpeg" height="200" /></div>

</details>

<summary><b>思路</b></summary>

- 棋盘搜索，非常典型的 DFS + 回溯问题；

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<details><summary><b>Python：DFS + 回溯</b></summary>

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if len(board) < 1:
            return False

        m, n = len(board), len(board[0])

        # 使用内部函数，可以减少一些参数的传递，同时比成员方法更简洁
        def dfs(i, j, k):  # i, j, k 分别表示 board[i][j] 和 word[k]
            if not 0 <= i < m or not 0 <= j < n:  # 先判断是否越界
                return False

            if board[i][j] != word[k]:  # 这一步可以合并到越界判断，但会损失一些可读性，故分离出来单独判断
                return False
            else:  # board[i][j] == word[k]:  # 如果当前位置字符相同，继续深度搜索
                if k == len(word) - 1:  # 如果字符已经全部匹配成功，返回 True
                    return True

                # 置空，表示该位置已访问过；一些代码中会使用一个新的矩阵记录位置是否访问，这里直接在原矩阵上标记
                board[i][j] = ''
                # 继续遍历 4 个方向
                flag = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
                # 这一步是容易忽略的：因为需要回溯，所以必须还原该位置的元素
                board[i][j] = word[k]

                return flag

        # board 中每一个位置都可能是起始位置，所以要循环遍历
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
```

</details>

---

### `剑指Offer 1300 机器人的运动范围 (中等, 2021-11)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["DFS"],
    "来源": "剑指Offer",
    "编号": "1300",
    "难度": "中等",
    "标题": "机器人的运动范围"
}-->

<summary><b>问题描述</b></summary>

```txt
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

示例 1：
    输入：m = 2, n = 3, k = 1
    输出：3
示例 2：
    输入：m = 3, n = 1, k = 0
    输出：1

提示：
    1 <= n,m <= 100
    0 <= k <= 20

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<summary><b>思路</b></summary>

- 本题也可以使用广度优先搜索；
  > [机器人的运动范围（ 回溯算法，DFS / BFS ，清晰图解）](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/solution/mian-shi-ti-13-ji-qi-ren-de-yun-dong-fan-wei-dfs-b/)

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<details><summary><b>Python：DFS+回溯</b></summary>

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:

        def dig_sum(x):  # 求数位之和
            s = 0
            while x != 0:
                s += x % 10
                x = x // 10
            return s

        def dfs(i, j):
            if not 0 <= i < m or not 0 <= j < n or dig_sum(i) + dig_sum(j) > k:
                return 0

            if (i, j) in visited:  # 如果已经访问过
                return 0
            else:
                visited.add((i, j))  # 访问标记
                return 1 + dfs(i + 1, j) + dfs(i, j + 1)  # 因为只能往左或往右，所以只需要搜索两个方向

        visited = set()
        return dfs(0, 0)
```

</details>

---

### `剑指Offer 1700 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["DFS"],
    "来源": "剑指Offer",
    "编号": "1700",
    "难度": "中等",
    "标题": "打印从1到最大的n位数（N叉树的遍历）"
}-->

<summary><b>问题简述</b></summary>

```txt
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数（考虑大数情况）；
比如输入 3，则打印出 1~999
```

<details><summary><b>详细描述</b></summary>

```txt
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

示例 1:
    输入: n = 1
    输出: [1,2,3,4,5,6,7,8,9]

说明：
    用返回一个整数列表来代替打印
    n 为正整数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<summary><b>思路</b></summary>

- 考虑大数情况下，直接遍历会存在越界问题；
- 本题实际上是一个 N 叉树（N=9）的遍历问题；

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<details><summary><b>Python：不考虑大数</b></summary>

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        res = []
        for i in range(1, 10 ** n):
            res.append(i)
        return res
```

</details>

<details><summary><b>Python：考虑大数</b></summary>

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:

        ret = []
        dig = '0123456789'
        buf = [''] * n

        def process(buf):
            """去除前置0"""
            start = 0
            while start < n - 1 and buf[start] == '0':  # 保留至少一个 0
                start += 1
            return int(''.join(buf[start:]))  # LeetCode要求返回 int

        def dfs(k):
            """DFS全排列"""
            if k == n:
                ret.append(process(buf))
                return

            for i in dig:  # 每一位都有 0-9 10种取法
                buf[k] = i
                dfs(k+1)

        dfs(0)
        return ret[1:]  # 要求从 1 开始，故移除第一位
```

</details>

---

### `剑指Offer 3400 二叉树中和为某一值的路径 (中等, 2021-12)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["二叉树", "DFS"],
    "来源": "剑指Offer",
    "编号": "3400",
    "难度": "中等",
    "标题": "二叉树中和为某一值的路径"
}-->

<summary><b>问题简述</b></summary>

```txt
给定二叉树 root 和一个整数 targetSum ，找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
```

<details><summary><b>详细描述</b></summary>

```txt
给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

示例 1：
    输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
    输出：[[5,4,11,2],[5,8,4,5]]
示例 2：
    输入：root = [1,2,3], targetSum = 5
    输出：[]
示例 3：
    输入：root = [1,2], targetSum = 0
    输出：[]

提示：
    树中节点总数在范围 [0, 5000] 内
    -1000 <= Node.val <= 1000
    -1000 <= targetSum <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 先序深度优先搜索；
- 因为要保存路径，所以还要加上回溯序列；

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        if not root: return []

        ret = []
        buf = []
        def dfs(R, T):
            # 这样写会导致结果输出两次，原因是如果当前叶节点满足后，会继续遍历其左右两个空节点，导致结果被添加两次
            # if not R:
            #     if T == 0:
            #         ret.append(buf[:])
            #     return

            if not R: return
            if R.left is None and R.right is None:
                if T == R.val:
                    ret.append(buf[:] + [R.val])  # 直接传 buf 会有问题，而 buf[:] 相对于 buf 的一份浅拷贝
                return

            buf.append(R.val)
            dfs(R.left, T - R.val)
            dfs(R.right, T - R.val)
            buf.pop()
        
        dfs(root, target)
        return ret
```

</details>

---

### `剑指Offer 3800 字符串的排列（全排列） (中等, 2021-12)`

[![DFS+剪枝](https://img.shields.io/badge/DFS+剪枝-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![经典](https://img.shields.io/badge/经典-lightgray.svg)](基础-经典问题&代码.md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["DFS+剪枝", "经典"],
    "来源": "剑指Offer",
    "编号": "3800",
    "难度": "中等",
    "标题": "字符串的排列（全排列）"
}-->

<summary><b>问题简述</b></summary>

```txt
输入一个字符串，打印出该字符串中字符的所有排列。
```

<details><summary><b>详细描述</b></summary>

```txt
输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

示例:
    输入：s = "abc"
    输出：["abc","acb","bac","bca","cab","cba"]

限制：
    1 <= s 的长度 <= 8

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：DFS树状遍历+剪枝</b></summary>

- 深度优先求全排列的过程实际上相当于是一个**多叉树的先序遍历过程**；
    - 假设共有 `n` 种状态都不重复，则：
    - 第一层有 `n` 种选择；
    - 第二层有 `n - 1` 种选择；
    - ...
    - 共有 `n!` 种可能；

    <details><summary><b>图示</b></summary>

    <div align="center"><img src="../_assets/剑指Offer_0038_中等_字符串的排列2.png" height="200" /></div>

    </details>

**本题的难点是如何过滤重复的状态**

- **写法1）** 遍历所有状态，直接用 `set` 保存结果（不剪枝）：

    <details><summary><b>Python</b></summary>

    ```python
    class Solution:
        def permutation(self, s: str) -> List[str]:

            N = len(s)
            buf = []
            ret = set()
            visited = [False] * N
            def dfs(deep):
                if deep == N:
                    ret.add(''.join(buf))
                    return

                for i in range(N):
                    if not visited[i]:
                        # 标记
                        buf.append(s[i])
                        visited[i] = True
                        # 进入下一层
                        dfs(deep + 1)
                        # 回溯（撤销标记）
                        buf.pop()  
                        visited[i] = False
            
            dfs(0)
            return list(ret)
    ```

    </details>

- **写法2）** 跳过重复字符（需排序）：
    - 其中用于剪枝的代码不太好理解，其解析详见：[「代码随想录」剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/dai-ma-sui-xiang-lu-jian-zhi-offer-38-zi-gwt6/)
  
        ```python
        if not visited[i - 1] and i > 0 and s[i] == s[i - 1]: 
            continue
        ```

    <details><summary><b>Python</b></summary>

    ```python
    class Solution:
        def permutation(self, s: str) -> List[str]:

            s = sorted(s)  # 排序，使相同字符在一起
            N = len(s)
            ret = []  # 保存结果
            buf = []  # 临时结果
            visited = [False] * N  # 记录是否访问
            def dfs(deep):  # 传入递归深度
                if deep == N:
                    ret.append(''.join(buf))
                    return

                for i in range(N):
                    # 剪枝
                    if visited[i - 1] is False and i > 0 and s[i] == s[i - 1]:
                        continue

                    # 下面的代码居然可以（区别仅在于 visited[i - 1] 的状态），
                    # 但是效率不如上面的，具体解析可参考：[「代码随想录」剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/dai-ma-sui-xiang-lu-jian-zhi-offer-38-zi-gwt6/)
                    # if visited[i - 1] is True and i > 0 and s[i] == s[i - 1]:
                    #     continue

                    if not visited[i]:  # 如果当前位置还没访问过
                        # 标记当前位置
                        visited[i] = True
                        buf.append(s[i])
                        # 下一个位置
                        dfs(deep + 1)
                        # 回溯
                        buf.pop()
                        visited[i] = False

            dfs(0)
            return ret
    ```

    </details>

- **写法3）** 在每一层用一个 `set` 保存已经用过的字符（不排序）：

    <details><summary><b>Python</b></summary>

    ```python
    class Solution:
        def permutation(self, s: str) -> List[str]:

            N = len(s)
            buf = []
            ret = set()
            visited = [False] * N
            def dfs(deep):
                if deep == N:
                    ret.add(''.join(buf))
                    return

                used = set()  # 记录用过的字符
                for i in range(N):
                    if s[i] in used:  # 如果是已经用过的
                        continue

                    if not visited[i]:
                        # 标记
                        used.add(s[i])
                        buf.append(s[i])
                        visited[i] = True
                        # 进入下一层
                        dfs(deep + 1)
                        # 回溯（撤销标记）
                        buf.pop()  
                        visited[i] = False
            
            dfs(0)
            return list(ret)
    ```

    </details>

- **写法2）** 原地交换
    > [剑指 Offer 38. 字符串的排列（回溯法，清晰图解）](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/mian-shi-ti-38-zi-fu-chuan-de-pai-lie-hui-su-fa-by/)

    - 这个写法有点像“下一个排列”，只是没有使用字典序；

    <details><summary><b>Python</b></summary>

    ```python
    class Solution:
        def permutation(self, s: str) -> List[str]:
            N = len(s)
            buf = list(s)
            ret = []

            def dfs(deep):
                if deep == N - 1:
                    ret.append(''.join(buf))   # 添加排列方案
                    return

                used = set()
                for i in range(deep, N):  # 注意遍历范围，类似选择排序
                    if buf[i] in used:  # 已经用过的状态
                        continue

                    used.add(buf[i])
                    buf[deep], buf[i] = buf[i], buf[deep]  # 交换，将 buf[i] 固定在第 deep 位
                    dfs(deep + 1)               # 开启固定第 x + 1 位字符
                    buf[deep], buf[i] = buf[i], buf[deep]  # 恢复交换

            dfs(0)
            return ret
    ```

    </details>


<summary><b>思路2：下一个排列</b></summary>

> [字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/solution/zi-fu-chuan-de-pai-lie-by-leetcode-solut-hhvs/)

- 先排序得到最小的字典序结果；
- 循环直到不存在下一个更大的排列；

<details><summary><b>Python</b></summary>

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        
        def nextPermutation(nums: List[str]) -> bool:
            i = len(nums) - 2
            while i >= 0 and nums[i] >= nums[i + 1]:
                i -= 1

            if i < 0:
                return False
            else:
                j = len(nums) - 1
                while j >= 0 and nums[i] >= nums[j]:
                    j -= 1
                nums[i], nums[j] = nums[j], nums[i]

            left, right = i + 1, len(nums) - 1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

            return True

        buf = sorted(s)
        ret = [''.join(buf)]
        while nextPermutation(buf):
            ret.append(''.join(buf))

        return ret
```

</details>

---

### `剑指Offer 5400 二叉搜索树的第k大节点 (简单, 2022-01)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![dfs](https://img.shields.io/badge/dfs-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](合集-剑指Offer.md)

<!--{
    "tags": ["二叉树", "dfs"],
    "来源": "剑指Offer",
    "编号": "5400",
    "难度": "简单",
    "标题": "二叉搜索树的第k大节点"
}-->

<summary><b>问题简述</b></summary>

```txt
给定一棵二叉搜索树，请找出其中第 k 大的节点的值。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一棵二叉搜索树，请找出其中第 k 大的节点的值。

示例 1:
    输入: root = [3,1,4,null,2], k = 1
       3
      / \
     1   4
      \
       2
    输出: 4
示例 2:
    输入: root = [5,3,6,2,4,null,null,1], k = 3
           5
          / \
         3   6
        / \
       2   4
      /
     1
    输出: 4

限制：
    1 ≤ k ≤ 二叉搜索树元素个数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<!-- <div align="center"><img src="./_assets/xxx.png" height="300" /></div> -->

</details>


<summary><b>思路</b></summary>

- 根据二叉搜索树的性质，其中序遍历的结果为递增序列；
- 为了得到第 k 大的数，需要递减序列，“反向”中序遍历即可：即按“右中左”的顺序深度搜索（正向为“左中右”）；
- 利用辅助变量提前结束搜索；


<details><summary><b>C++</b></summary>

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    int k;
    int ret;

    void inOrder(TreeNode* node) {
        if (node == nullptr) return;

        inOrder(node->right);  // 先遍历右子树
        if (--this->k == 0) {  // 因为 k>0，实际上第 1 大指的是索引为 0 的位置，所以要先 --
            this->ret = node->val;
            return;
        }
        inOrder(node->left);
    }
    
public:
    int kthLargest(TreeNode* root, int k) {
        this->k = k;
        inOrder(root);
        return this->ret;
    }
};
```

</details>

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:

        self.cnt = 0
        self.ret = -1
        
        def dfs(node):
            if node is None:
                return 
            
            dfs(node.right)
            self.cnt += 1
            if self.cnt == k:
                self.ret = node.val
                return 
            dfs(node.left)
        
        dfs(root)
        return self.ret
```

</details>

---

### `牛客 0005 二叉树根节点到叶子节点的所有路径和 (中等, 2022-01)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![牛客](https://img.shields.io/badge/牛客-lightgray.svg)](合集-牛客.md)

<!--{
    "tags": ["二叉树", "DFS"],
    "来源": "牛客",
    "难度": "中等",
    "编号": "0005",
    "标题": "二叉树根节点到叶子节点的所有路径和",
    "公司": ["小米", "快手", "字节"]
}-->

<summary><b>问题简述</b></summary>

```txt
给定二叉树，求所有路径和，路径定义如下：
假设某条路径的从根节点到叶节点的值为 1->2->3，则记该条路径表示的值为 123；
输入确保每个节点的值在 0~9 之间；

示例
    1
   2 3
结果：25（12+13=25）
```
> [二叉树根节点到叶子节点的所有路径和_牛客题霸_牛客网](https://www.nowcoder.com/practice/185a87cd29eb42049132aed873273e83)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：DFS</b></summary>

<details><summary><b>Python</b></summary>

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param root TreeNode类 
# @return int整型
#
class Solution:
    def sumNumbers(self , root: TreeNode) -> int:
        # write code here
        
        self.ret = 0
        
        def dfs(node: TreeNode, sum_):
            if not node:
                # self.ret += sum_  # 放在这里会导致“加两次”
                return
            
            sum_ = sum_ * 10 + node.val
            if not node.left and not node.right:
                self.ret += sum_
                return
            
            if node.left:
                dfs(node.left, sum_)
            if node.right:
                dfs(node.right, sum_)
        
        dfs(root, 0)
        return self.ret
```

</details>

---

### `牛客 0008 二叉树中和为某一值的路径(二) (中等, 2022-01)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![牛客](https://img.shields.io/badge/牛客-lightgray.svg)](合集-牛客.md)

<!--{
    "tags": ["二叉树", "DFS"],
    "来源": "牛客",
    "难度": "中等",
    "编号": "0008",
    "标题": "二叉树中和为某一值的路径(二)",
    "公司": []
}-->

<summary><b>问题简述</b></summary>

```txt
给定二叉树 root 和目标值 target，返回所有和为 target 的路径。
规定路径必须从根节点开始到叶子节点。
```
> [二叉树中和为某一值的路径(二)_牛客题霸_牛客网](https://www.nowcoder.com/practice/b736e784e3e34731af99065031301bca)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：DFS</b></summary>

<details><summary><b>Python</b></summary>

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param root TreeNode类 
# @param target int整型 
# @return int整型二维数组
#
class Solution:
    def FindPath(self , root: TreeNode, target: int) -> List[List[int]]:
        # write code here
        
        ret = []
        tmp = []
        
        def dfs(node, k):
            if not node: return
            
            tmp.append(node.val)
            k -= node.val
            
            if not node.left and not node.right and k == 0:
                ret.append(tmp[:])
            
            dfs(node.left, k)
            dfs(node.right, k)
            k += node.val
            tmp.pop()
        
        dfs(root, target)
        return ret
```

</details>

---

### `牛客 0009 二叉树中和为某一值的路径(一) (简单, 2022-01)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![牛客](https://img.shields.io/badge/牛客-lightgray.svg)](合集-牛客.md)

<!--{
    "tags": ["二叉树", "DFS"],
    "来源": "牛客",
    "难度": "简单",
    "编号": "0009",
    "标题": "二叉树中和为某一值的路径(一)",
    "公司": ["腾讯", "字节", "京东"]
}-->

<summary><b>问题简述</b></summary>

```txt
给定二叉树 root 和目标值 target，判断是否存在路径和等于 target。
规定路径必须从根节点开始到叶子节点。
```
> [二叉树中和为某一值的路径(一)_牛客题霸_牛客网](https://www.nowcoder.com/practice/508378c0823c423baa723ce448cbfd0c)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：DFS</b></summary>

<details><summary><b>Python</b></summary>

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param root TreeNode类 
# @param sum int整型 
# @return bool布尔型
#
class Solution:
    def hasPathSum(self , root: TreeNode, sum: int) -> bool:
        # write code here
        
        def dfs(node, k):
            if not node: return False
            
            if k == node.val and not node.left and not node.right:
                return True
            
            return dfs(node.left, k - node.val) or dfs(node.right, k - node.val)
        
        return dfs(root, sum)
```

</details>

---

### `牛客 0020 数字字符串转化成IP地址 (中等, 2022-01)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![牛客](https://img.shields.io/badge/牛客-lightgray.svg)](合集-牛客.md)

<!--{
    "tags": ["DFS"],
    "来源": "牛客",
    "难度": "中等",
    "编号": "0020",
    "标题": "数字字符串转化成IP地址",
    "公司": ["百度", "字节", "快手"]
}-->

<summary><b>问题简述</b></summary>

```txt
给定只包含数字的字符串，将该字符串转化成 IP 地址的形式，返回所有可能的情况。
例如：给出的字符串为"25525522135",
返回：["255.255.22.135", "255.255.221.35"]
```
> [数字字符串转化成IP地址_牛客题霸_牛客网](https://www.nowcoder.com/practice/ce73540d47374dbe85b3125f57727e1e)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：DFS + 回溯</b></summary>

- 相当于构建一颗三叉树；

<div align="center"><img src="../_assets/牛客0020.png" height="300" /></div>

<details><summary><b>Python</b></summary>

```python
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param s string字符串 
# @return string字符串一维数组
#
class Solution:
    def restoreIpAddresses(self , s: str) -> List[str]:
        # write code here
        
        ret = []
        
        def valid(x):
            """验证函数"""
            if not x:  # 非空
                return False
            if x.startswith('0') and len(x) > 1:  # 存在前缀 0
                return False
            return int(x) <= 255
        
        def dfs(k, depth, tmp):
            if depth == 3:  # 到第三层的时候，直接判断所有剩余字符
                if valid(s[k:]):
                    tmp.append(s[k:])
                    ret.append('.'.join(tmp))
                    tmp.pop()  # 这里也要回溯
                return
            
            for i in range(1, 4):
                sub = s[k: k + i]
                if valid(sub):
                    tmp.append(sub)
                    dfs(k + i, depth + 1, tmp)
                    tmp.pop()
        
        dfs(0, 0, [])
        return ret
```

</details>

---

### `牛客 0045 实现二叉树先序、中序、后序遍历 (中等, 2022-03)`

[![DFS](https://img.shields.io/badge/DFS-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![牛客](https://img.shields.io/badge/牛客-lightgray.svg)](合集-牛客.md)

<!--{
    "tags": ["DFS"],
    "来源": "牛客",
    "难度": "中等",
    "编号": "0045",
    "标题": "实现二叉树先序、中序、后序遍历",
    "公司": []
}-->

<summary><b>问题简述</b></summary>

```txt
给定一棵二叉树，分别按照二叉树先序，中序和后序打印所有的节点。
```
> [实现二叉树先序，中序和后序遍历_牛客题霸_牛客网](https://www.nowcoder.com/practice/a9fec6c46a684ad5a3abd4e365a9d362)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：DFS</b></summary>

<details><summary><b>Python</b></summary>

```python
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 
# @param root TreeNode类 the root of binary tree
# @return int整型二维数组
#
class Solution:
    def threeOrders(self , root: TreeNode) -> List[List[int]]:
        # write code here
        
        p, i, s = [], [], []
        
        def dfs(x):
            if x is None:
                return 
            
            p.append(x.val)
            dfs(x.left)
            i.append(x.val)
            dfs(x.right)
            s.append(x.val)
        
        dfs(root)
        return [p, i, s]
```

</details>

---
