<!-- Tag: TreeDP -->

> 以下经验总结自[《左程云算法教程》](https://www.bilibili.com/video/BV1NU4y1M7rF?p=47)（P47）


### 使用场景
- 需要自底向上（后序遍历）解决的二叉树问题或递归问题；
    > 实际上就是在树结构上进行状态转移，即树形 DP 问题；

### 技巧描述

1. 考虑为了计算出以 X 为头结点的答案，需要从左右子树获得哪些信息；
    > 这一步是需要主动思考和经验积累的；
2. 假设这些信息是已知的，使用这些信息计算出 X 节点下同样的信息并返回；常见的集中情形：
    1. 答案一定包含 X 节点
        > [110. 平衡二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/balanced-binary-tree/submissions/)
    2. 答案不一定经过 X 节点，此时需要根据答案是否包含 X 分情况讨论
        > [124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)  
        > [543. 二叉树的直径 - 力扣（LeetCode）](https://leetcode-cn.com/problems/diameter-of-binary-tree/submissions/)
    3. 更复杂的分情况讨论
        > [968. 监控二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-cameras/)
3. 考虑空节点如何构造这些信息（递归基），一般分两种情况：
    1. 空节点有定义，那么就按照定义赋值；
    2. 空节点无定义，则返回 None，然后在使用左右子树的信息时做非空判断；

> 其实不光是二叉树，所有需要自底向上递归的问题都可以遵从上述原则；因为二叉树是最基础的递归结构，在应用这个技巧的时候最直观；

> **模板或者技巧只是帮我们解决了怎么组织代码的问题，而如何写出正确的代码？一靠观察，二靠积累；**


#### 示例1：判断是否为平衡二叉树
> [110. 平衡二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/balanced-binary-tree/submissions/)

1. 为了确定 X 是否为平衡二叉树，需要从左右子树知道的信息：
    1. 子树本身是否平衡（`is_balanced: bool`）
    2. 子树的高度（`height: int`）
2. 计算 X 节点的上述信息：
    ```python
    # 已知左右子树的信息 l 和 r
    # 则 X 的 is_balanced 和 height 分别为
    x.is_balanced = abs(l.height - r.height) <= 1 and l.is_balanced and r.is_balanced
    x.height = max(l.height, r.height) + 1
    ```
3. 空节点的定义：
    ```python
    is_balanced = True
    height = 0
    ```

<details><summary><b>完整代码（Python）</b></summary>

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
            
            # 假设已知左右子树的信息
            l, r = dfs(x.left), dfs(x.right)
            # 利用左右子树的信息计算 X 的信息
            is_balanced = abs(l.height - r.height) <= 1 and l.is_balanced and r.is_balanced
            height = max(l.height, r.height) + 1
            # 返回 X 的信息
            return Info(is_balanced, height)
        
        return dfs(root).is_balanced  # 返回需要的信息
```

</details>


<!-- #### 示例2：最大二叉搜索子树
> [333. 最大 BST 子树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/largest-bst-subtree/)

1. 为了找到 X 的最大二叉搜索子树，需要从左右子树知道的信息：
    1. 子树是否为二叉搜索树（`is_bst: bool`）
    2. 子树中的最小值（`min: int`）
    3. 子树中的最大值（`max: int`）
2. 计算 X 节点的上述信息：
    1. 


<details><summary><b>完整代码（Python）</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxBST(self, root: Optional[TreeNode]) -> int:
        
        from collections import namedtuple

        # max: 该节点能提供的最大路径（含节点本身）
        # ret: 该节点下的最大路径（可能不包含该节点）
        Info = namedtuple('Info', ['is_bst', 'max', 'min'])

        self.ret = None
        
        def dfs(x):
            if not x:
                # 对空节点，初始化 min=inf, max=-inf
                return Info(True, float('inf'), float('-inf'))
            
            l, r = dfs(x.left), dfs(x.right)
            is_bst = l.is_bst and r.is_bst and l.max < x.val < r.min
            x_min = max(x.val, l.min)
            x_max = max(x.val, r.max)

            if is_bst:
                self.ret = x
            return Info(is_bst, x_min, x_max)
        
        dfs(root)
        return self.ret
```

</details>
 -->


#### 示例2：二叉树中的最大路径和
> [124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

1. 为了确定 X 是否为平衡二叉树，需要从左右子树知道的信息：
    1. 子树能提供的最大路径和（包含子节点），记 `h: int`
        > 即以该子节点为终点的最大路径和
    2. 子树的最大路径和（不一定包含子节点），记 `s: int`
        > 即原问题的子问题，该路径不一定会经过子节点
2. 计算 X 节点的这些信息：
    1. `x_h = x.val + max(0, l.h, r.h)`
        > 以 X 为终点，所以必须算上 X 的值，同时如果子树 h 为负就舍去；
    2. `x_s = max(l.s, r.s, max(l.h, 0) + max(r.h, 0) + x.val)`
        > 分情况讨论，不经过 X 和经过 X 的路径，取最大值；
3. 空节点：
    1. `h=0`
    2. `s=-inf`
        > 为什么空节点的 s 要初始化为负无穷？因为至少要经过一个节点。如果初始化为 0，那么当树中的值都是负数时，就会出错；

<details><summary><b>完整代码（Python）</b></summary>

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


<!-- 
> **一些感想**：
>> 刚开始看到这个技巧的时候，我觉得这不是显然的吗，这算什么技巧？当看到具体 coding 后，才明白同样的思路不代表相同的代码；

>> 显然的东西真的显然吗？沐神最近的这个视频也提到了，很有可能是在你听到了这个想法之后才觉得它特别显然，但是在你没听到之前，你就是想不到。这里也是类似的，递归的定义都知道，但就是写不出这么简洁的代码。总结一下，就是**用进废退**，**学无止境**。
>>> [你（被）吐槽过论文不够 novel 吗？【论文精读】_李沐](https://www.bilibili.com/video/BV1ea41127Bq?spm_id_from=333.851.dynamic.content.click)  
-->


### 经典问题

- 【简单】[110. 平衡二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/balanced-binary-tree/)
- 【简单（中等）】[543. 二叉树的直径 - 力扣（LeetCode）](https://leetcode-cn.com/problems/diameter-of-binary-tree/submissions/)
- 【困难（中等）】[124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- 【中等，会员】[333. 最大 BST 子树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/largest-bst-subtree/)
- 【困难】[968. 监控二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/binary-tree-cameras/)