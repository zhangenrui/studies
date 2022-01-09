<!-- Tag: 二叉树 -->

<summary><b>问题简述</b></summary>

```txt
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树

            3
          /   \
         5     1
        / \   / \
       6   2 0   8
          / \
         7   4

示例 1:
    输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
    输出: 3
    解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
示例 2:
    输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
    输出: 5
    解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。

说明:
    所有节点的值都是唯一的。
    p、q 为不同节点且均存在于给定的二叉树中。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 因为必须先找到目标节点才能确定路线，所以要后序遍历；
- 当找到目标节点时，返回 flag，指示上级节点是否为祖先节点；

<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

        # 后序遍历搜索历史祖先，因为是后序遍历，所以 trace 是倒序的
        def dfs(node, target, trace):
            if node is None:
                return False
            if node.val == target.val:
                trace.append(node)  # 根据定义，自己也是自己的祖先节点
                return True
            
            if dfs(node.left, target, trace) or dfs(node.right, target, trace):
                trace.append(node)
                return True
            else:
                return False
        
        # 分别找出 p 和 q 的祖先路径
        trace_p = []
        dfs(root, p, trace_p)
        # print(trace_p)
        trace_q = []
        dfs(root, q, trace_q)
        # print(trace_q)

        # 遍历找出最后一个相同的祖先
        ret = None
        for l, r in zip(trace_p[::-1], trace_q[::-1]):
            if l.val == r.val:
                ret = l
            else:
                break
        
        return ret
```

</details>


**优化**：不使用额外空间存储祖先路径，即在遍历过程中判断；
> [二叉树的最近公共祖先（DFS ，清晰图解） - Krahets](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/solution/mian-shi-ti-68-ii-er-cha-shu-de-zui-jin-gong-gon-7/)

- 如果 node 仅是 p 和 q 的公共祖先（但不是最近公共祖先），那么 node 的左右子树之一必也是 p 和 q 的公共祖先；
- 如果 node 是 p 和 q 的最近公共祖先，那么 node 的左右子树都不是 p 和 q 的公共祖先；
- 根据以上两条性质，可知，如果 node 是 p、q 的**最近公共祖先**，有：
    - node 是 p、q 的公共祖先，且 p 和 q 分别在 node 的两侧；
    - node 是 p 或 q 之一，且是另一个节点的祖先；


<details><summary><b>Python</b></summary>

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        
        def dfs(node):
            # 下面两个判断条件可以写在一起，为了使逻辑更清晰，故分开写
            if node is None:  # 说明当前路径上没有 p 或 q
                return None
            if node == p or node == q:  # 说明当前路径上存在 p 或 q
                return node
            
            l = dfs(node.left)
            r = dfs(node.right)

            # 返回的非 None 节点都是 p 和 q 的公共祖先
            if l is None and r is not None:  # r 是 p 和 q 之一，且是另一个节点的祖先
                return r
            elif r is None and l is not None:  # l 是 p 和 q 之一，且是另一个节点的祖先
                return l
            elif l and r:  # p 和 q 分别在 node 的两侧
                return node
            else:
                return None

        return dfs(root)

```

</details>