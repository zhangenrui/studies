# 二叉树自底向上的递归技巧（树形DP）

<!-- Tag: TreeDP -->

> 笔记详见：[**二叉树自底向上的递归技巧（树形DP）**](../../notes/算法/二叉树自底向上的递归技巧（树形DP）)

Problems
---
- [`LeetCode 0110 平衡二叉树 (简单, 2022-02)`](#leetcode-0110-平衡二叉树-简单-2022-02)
- [`LeetCode 0124 二叉树中的最大路径和 (困难, 2022-02)`](#leetcode-0124-二叉树中的最大路径和-困难-2022-02)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](#leetcode-0437-路径总和iii-中等-2022-02)

---

### `LeetCode 0110 平衡二叉树 (简单, 2022-02)`

[![TreeDP](https://img.shields.io/badge/TreeDP-lightgray.svg)](技巧-二叉树自底向上的递归技巧（树形DP）.md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](合集-LeetCode.md)

<!--{
    "tags": ["TreeDP"],
    "来源": "LeetCode",
    "难度": "简单",
    "编号": "0110",
    "标题": "平衡二叉树",
    "公司": []
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
```
> [110. 平衡二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/balanced-binary-tree/)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：树形DP</b></summary>

- 考虑对以 X 为头结点的树，为了确定其是否为平衡二叉树，需要从左右子树获取哪些信息？
- 根据定义，显然需要知道两个信息：
    1. 子树是否为平衡二叉树（`is_balanced: bool`）；
    2. 子树的高度（`height: int`）；
- 假设子树的这些信息已知，怎么求 X 节点的上述信息：
    1. `x_is_balanced = l.is_balanced and r.is_balanced and abs(l.height - r.height) <= 1`
        > 即左右子树都平衡，且高度差小于等于 1
    2. `x_height = max(l.height, r.height) + 1`
- 对空节点，有：
    ```python
    is_balanced = True
    height = 0
    ```

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        from collections import namedtuple

        # 用一个结构来组织需要的信息，可以直接用 tuple，这里是为了更直观
        Info = namedtuple('Info', ['is_balanced', 'height'])

        def dfs(x):
            if not x:  # 空节点
                return Info(True, 0)
            
            l, r = dfs(x.left), dfs(x.right)
            is_balanced = l.is_balanced and r.is_balanced and abs(l.height - r.height) <= 1
            height = max(l.height, r.height) + 1
            return Info(is_balanced, height)
        
        return dfs(root).is_balanced  # 返回需要的信息
```

</details>

---

### `LeetCode 0124 二叉树中的最大路径和 (困难, 2022-02)`

[![TreeDP](https://img.shields.io/badge/TreeDP-lightgray.svg)](技巧-二叉树自底向上的递归技巧（树形DP）.md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](合集-LeetCode.md)

<!--{
    "tags": ["TreeDP"],
    "来源": "LeetCode",
    "难度": "困难",
    "编号": "0124",
    "标题": "二叉树中的最大路径和",
    "公司": []
}-->

<summary><b>问题简述</b></summary>

```txt
路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。
```
> [124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```
-->

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：树形DP</b></summary>

- 考虑对以 X 为头结点的树，为了求得最大路径和，需要从左右子树获取哪些信息？
- 根据要求，显然需要知道：
    1. 子树能提供的最大和路径，即以子树节点为终点的最大路径，记 `h: int`；
        > 为什么需要这个信息：**为了计算经过 X 节点的最大路径**，这条路径的特点是：从某一节点出发到达子树节点，经过 X 节点后，再进入另一个子节点的最大和路径；
        >> 这是本题除 coding 外最大的难点，能想出这个信息也就基本解决这个问题了；
    2. 子树中的最大路径和（即子问题）：为了比较得出全局最大路径和，记 `s: int`；
- 假设子树的这些信息已知，怎么求 X 节点的上述信息：
    1. `x_h = x.val + max(0, l.h, r.h)`
        > 因为需要经过 X 节点，所以必须加上 x.val，同时如果左右子树提供的 h 小于 0，那么不如舍去；
    2. `x_s = max(l.s, r.s, max(l.h, 0) + max(r.h, 0) + x.val)`
        > 这一步容易写成 `x_s = max(l.s, r.s, x_h)` 或者 `x_s = max(l.s, r.s, l.h + r.h + x.val)`，都是对问题理解不到位；
    > 重申：模板只是帮我们解决怎么组织代码的问题，而写出正确的代码一靠观察，二靠积累；
- 对空节点，有：
    ```python
    h = 0
    s = -inf  # 因为题目要求至少包含一个节点，如果设为 0，那么当所有节点为负数时，就会出错
    ```

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        from dataclasses import dataclass

        @dataclass
        class Info:
            h: int  # 该节点能提供的最大路径（含节点本身）
            s: int  # 该节点下的最大路径（可能不包含该节点）

        # 事实上 Info 里的 s 完全可以用一个全局变量来代替，这里是为了尽量拟合模板；熟练之后就不必这么做了。
        
        def dfs(x):
            if not x:
                # 对空节点，初始化 h=0, s=负无穷
                return Info(0, float('-inf'))
            
            l, r = dfs(x.left), dfs(x.right)
            x_h = x.val + max(0, l.h, r.h)
            x_s = max(l.s, r.s, max(l.h, 0) + max(r.h, 0) + x.val)
            return Info(x_h, x_s)
        
        return dfs(root).s
```

</details>

---

### `LeetCode 0437 路径总和III (中等, 2022-02)`

[![二叉树](https://img.shields.io/badge/二叉树-lightgray.svg)](数据结构-二叉树.md)
[![深度优先搜索](https://img.shields.io/badge/深度优先搜索-lightgray.svg)](算法-深度优先搜索(DFS).md)
[![前缀和](https://img.shields.io/badge/前缀和-lightgray.svg)](技巧-前缀和.md)
[![TreeDP](https://img.shields.io/badge/TreeDP-lightgray.svg)](技巧-二叉树自底向上的递归技巧（树形DP）.md)
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
