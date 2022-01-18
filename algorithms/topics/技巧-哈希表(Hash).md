# 哈希表(Hash)

[Problems Index](#problems-index)

<!-- Tag: 哈希表、Hash -->

Problems Index
---
- [`LeetCode No.0001 两数之和 (简单, 2021-10)`](#leetcode-no0001-两数之和-简单-2021-10)
- [`LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`](#leetcode-no0187-重复的dna序列-中等-2021-10)
- [`剑指Offer No.0300 数组中重复的数字 (简单, 2021-11)`](#剑指offer-no0300-数组中重复的数字-简单-2021-11)
- [`剑指Offer No.3500 复杂链表的复制（深拷贝） (中等, 2021-12)`](#剑指offer-no3500-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer No.4800 最长不含重复字符的子字符串 (中等, 2021-12)`](#剑指offer-no4800-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer No.5000 第一个只出现一次的字符 (简单, 2021-12)`](#剑指offer-no5000-第一个只出现一次的字符-简单-2021-12)
- [`程序员面试金典 No.0102 判定是否互为字符重排 (简单, 2022-01)`](#程序员面试金典-no0102-判定是否互为字符重排-简单-2022-01)

---

### `LeetCode No.0001 两数之和 (简单, 2021-10)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](题集-LeetCode.md)

<!--{
    "tags": ["哈希表"],
    "来源": "LeetCode",
    "编号": "0001",
    "难度": "简单",
    "标题": "两数之和"
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
```


<details><summary><b>详细描述</b></summary>

```txt
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

示例 1：
    输入：nums = [2,7,11,15], target = 9
    输出：[0,1]
    解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
示例 2：
    输入：nums = [3,2,4], target = 6
    输出：[1,2]
示例 3：
    输入：nums = [3,3], target = 6
    输出：[0,1]
 

提示：
    2 <= nums.length <= 10^4
    -10^9 <= nums[i] <= 10^9
    -10^9 <= target <= 10^9
    只会存在一个有效答案

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/two-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<summary><b>思路</b></summary>

<details><summary><b>Python3</b></summary>

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:  # noqa
        """"""
        tmp = dict()

        for i in range(len(nums)):
            left = target - nums[i]  # 减去当前值
            if left in tmp:  # 如果差值在哈希表中，说明找到了答案
                return [tmp[left], i]

            tmp[nums[i]] = i  # 保存当前值的位置

        return []

```

</details>

---

### `LeetCode No.0187 重复的DNA序列 (中等, 2021-10)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![位运算](https://img.shields.io/badge/位运算-lightgray.svg)](技巧-位运算.md)
[![LeetCode](https://img.shields.io/badge/LeetCode-lightgray.svg)](题集-LeetCode.md)

<!--{
    "tags": ["哈希表", "位运算"],
    "来源": "LeetCode",
    "编号": "0187",
    "难度": "中等",
    "标题": "重复的DNA序列"
}-->

<summary><b>问题简述</b></summary>

```txt
找出由 ATCG 构成的字符串中所有重复且长度为 10 的子串；
```

<details><summary><b>详细描述</b></summary>

```txt
所有 DNA 都由一系列缩写为 'A'，'C'，'G' 和 'T' 的核苷酸组成，例如："ACGAATTCCG"。在研究 DNA 时，识别 DNA 中的重复序列有时会对研究非常有帮助。

编写一个函数来找出所有目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。

示例 1：
    输入：s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    输出：["AAAAACCCCC","CCCCCAAAAA"]
示例 2：
    输入：s = "AAAAAAAAAAAAA"
    输出：["AAAAAAAAAA"]

提示：
    0 <= s.length <= 10^5
    s[i] 为 'A'、'C'、'G' 或 'T'

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/repeated-dna-sequences
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<summary><b>思路</b></summary>

- 基本思路：哈希表计数；
- 如果直接使用子串本身作为哈希表的 key，那么时间复杂度和空间复杂度都是 `O(NL)`；而如果使用位运算+滑动窗口手动构造 key，可以把复杂度降为 `O(N)`；


<details><summary><b>子串作为 key</b></summary>

- 时间&空间复杂度：`O(NL)`；
```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        """"""
        # from collections import defaultdict
        L = 10

        cnt = defaultdict(int)
        ans = []
        for i in range(len(s) - L + 1):
            subs = s[i: i+L]
            cnt[subs] += 1
            if cnt[subs] == 2:
                ans.append(subs)

        return ans
```

</details>

<details><summary><b>位运算+滑动窗口</b></summary>

- 时间&空间复杂度：`O(N)`；
```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        """"""
        # from collections import defaultdict
        L = 10
        B = {'A': 0, 'T': 1, 'C': 2, 'G': 3}  # 分别为 00, 01, 10, 11

        if len(s) < L + 1:  # assert，否则部分用例会无法通过
            return []

        # 先计算前 9 位的值
        x = 0
        for i in range(L - 1):
            b = B[s[i]]
            x = (x << 2) | b

        ans = []
        cnt = defaultdict(int)
        for i in range(len(s) - L + 1):
            b = B[s[i + L - 1]]
            # 注意该有的括号不要少，避免运算优先级混乱
            x = ((x << 2) | b) & ((1 << (L * 2)) - 1)  # 滑动计算子串的 hash 值
            cnt[x] += 1
            if cnt[x] == 2:
                ans.append(s[i: i + L])

        return ans
```

</details>


<details><summary><b>位运算说明</b></summary>

- `(x << 2) | b`：
    ```python
    # 以为均为二进制表示
    设 x = 0010 1011, b = 10: 
    该运算相当于把 b “拼” 到 x 末尾

    x         :   0010 1011
    x = x << 2:   1010 1100
    
    x = x | b :   1010 1100
                | 0000 0010
                -----------
                  1010 1110
    ```
- `x & ((1 << (L * 2)) - 1)`
    ```python
    # 该运算把 x 除低 10 位前的所有位置置 0
    设 L = 5，x = 1110 1010 1010: 
    
    y = 1 << (L * 2):   0100 0000 0000
    y = y - 1       :   0011 1111 1111

    x = x & y       :   1110 1010 1010
                      & 0011 1111 1111
                      ----------------
                        0010 1010 1010

    ```

</details>

---

### `剑指Offer No.0300 数组中重复的数字 (简单, 2021-11)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](题集-剑指Offer.md)

<!--{
    "tags": ["哈希表"],
    "来源": "剑指Offer",
    "编号": "0300",
    "难度": "简单",
    "标题": "数组中重复的数字"
}-->

<summary><b>问题简述</b></summary>

```txt
找出数组中任意一个重复的数字。
```

<details><summary><b>详细描述</b></summary>

```txt
找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例 1：
    输入：
    [2, 3, 1, 0, 2, 5, 3]
    输出：2 或 3 

限制：
    2 <= n <= 100000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

</details>

<summary><b>思路</b></summary>

- 遍历数组，保存见过的数字，当遇到出现过的数字即返回


<details><summary><b>Python</b></summary>

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        tb = set()
        for i in nums:
            if i in tb:
                return i
            tb.add(i)
```

</details>

---

### `剑指Offer No.3500 复杂链表的复制（深拷贝） (中等, 2021-12)`

[![链表](https://img.shields.io/badge/链表-lightgray.svg)](数据结构-链表.md)
[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![经典](https://img.shields.io/badge/经典-lightgray.svg)](基础-经典问题&代码.md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](题集-剑指Offer.md)

<!--{
    "tags": ["链表", "哈希表", "经典"],
    "来源": "剑指Offer",
    "编号": "3500",
    "难度": "中等",
    "标题": "复杂链表的复制（深拷贝）"
}-->

<summary><b>问题简述</b></summary>

```txt
复制带随机指针的链表，返回复制后链表的头结点；
```

<details><summary><b>详细描述</b></summary>

**注意**：本题的输入输出带有迷惑性，它们并不是实际的输入和输出，而是链表的数组展现；

```txt
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

示例 1：
    输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
    输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：
    输入：head = [[1,1],[2,1]]
    输出：[[1,1],[2,1]]
示例 3：
    输入：head = [[3,null],[3,0],[3,null]]
    输出：[[3,null],[3,0],[3,null]]
示例 4：
    输入：head = []
    输出：[]
    解释：给定的链表为空（空指针），因此返回 null。

提示：
    -10000 <= Node.val <= 10000
    Node.random 为空（null）或指向链表中的节点。
    节点数目不超过 1000 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：哈希表</b></summary>

- 先看下普通链表的复制：

    <details><summary><b>普通链表的复制</b></summary>

    ```python
        class Solution:
            def copyList(self, head: 'Node') -> 'Node':
                cur = head
                ret = pre = Node(0)  # 伪头结点
                while cur:
                    node = Node(cur.val) # 复制节点 cur
                    pre.next = node      # 新链表的 前驱节点 -> 当前节点
                    # pre.random = '???' # 新链表的 「 前驱节点 -> 当前节点 」 无法确定
                    cur = cur.next       # 遍历下一节点
                    pre = node           # 保存当前新节点
                return ret.next
    ```

    </details>

- 首先要理解本题的难点：
    - 复制当前节点的时候，随机指针指向的节点可能还没有创建；
    - 即使你先按普通链表先把节点都创建出来，由于链表无法随机访问的性质，你也不知道随机节点在哪个位置；
- 解决方法是利用哈希表（写法1）：
    - 第一次遍历时，记录每个节点对应的复制节点；
    - 第二次遍历时，根据原链表的指向从哈希表中提取对应的节点，建立指向关系；
- 本题还有一种递归的写法（写法2）：
    - 同样用一个哈希表保存

<details><summary><b>Python：迭代（写法1）</b></summary>

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None  # 使用伪头结点，可以省去这行

        dp = dict()

        # 第一次遍历，生成复制节点，并记录到哈希表
        p = head
        while p:
            dp[p] = Node(p.val)
            p = p.next
        
        # 写法1：使用伪头结点，可以省去对 head 为 None 的判断
        cur = head
        ret = pre = Node(0)  # 伪头结点
        while cur:
            pre.next = dp[cur]  # 这里可以不用 get，因为一定存在
            pre.next.random = dp.get(cur.random)  # get 方法在 key 不存在时，默认返回 None
            cur = cur.next
            pre = pre.next

        return ret.next

        # 写法2：相比使用伪头结点
        # cur = head
        # while cur:
        #     dp[cur].next = dp.get(cur.next)
        #     dp[cur].random = dp.get(cur.random)
        #     cur = cur.next
        
        # return dp[head]
```

</details>


<details><summary><b>Python：递归（写法2）</b></summary>

- 【不推荐】虽然代码量会少一点，但是不好理解；

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None

        dp = dict()
        
        def dfs(p):
            if not p: return None

            if p not in dp:
                dp[p] = Node(p.val)
                dp[p].next = dfs(p.next)
                dp[p].random = dfs(p.random)
        
            return dp[p]
        
        return dfs(head)
```

</details>


<summary><b>思路2：复制+拆分</b></summary>

<div align="center"><img src="../_assets/剑指Offer_0035_中等_复杂链表的复制.png" height="300" /></div>

> 详见：[复杂链表的复制（哈希表 / 拼接与拆分，清晰图解）](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/solution/jian-zhi-offer-35-fu-za-lian-biao-de-fu-zhi-ha-xi-/)

- 注意这个方法需要遍历三次：
    - 第一次复制节点
    - 第二次设置随机节点
    - 第三次拆分
- 因为随机节点指向任意，所以必须先设置完所有随机节点后才能拆分；

<details><summary><b>Python</b></summary>

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None

        # 复制节点
        cur = head
        while cur:
            nod = Node(cur.val)  # 创建节点
            cur.next, nod.next = nod, cur.next  # 接入新节点
            cur = nod.next  # 遍历下一个节点

        # 设置随机节点，因为随机节点指向任意，所以必须先设置随机节点后才能断开
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next

        # 拆分节点
        cur = head
        ret = nxt = head.next
        while nxt.next:
            # 开始拆分
            cur.next = cur.next.next
            nxt.next = nxt.next.next

            # 下一组
            cur = cur.next
            nxt = nxt.next
        
        return ret
```

</details>

---

### `剑指Offer No.4800 最长不含重复字符的子字符串 (中等, 2021-12)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![双指针](https://img.shields.io/badge/双指针-lightgray.svg)](技巧-双指针.md)
[![动态规划](https://img.shields.io/badge/动态规划-lightgray.svg)](算法-动态规划(记忆化搜索)、递推.md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](题集-剑指Offer.md)

<!--{
    "tags": ["哈希表", "双指针", "动态规划"],
    "来源": "剑指Offer",
    "编号": "4800",
    "难度": "中等",
    "标题": "最长不含重复字符的子字符串"
}-->

<summary><b>问题简述</b></summary>

```txt
求字符串 s 中的最长不重复子串，返回其长度；
```

<details><summary><b>详细描述</b></summary>

```txt
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

示例 1:
    输入: "abcabcbb"
    输出: 3 
    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:
    输入: "bbbbb"
    输出: 1
    解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:
    输入: "pwwkew"
    输出: 3
    解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
        请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 
提示：
    s.length <= 40000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：双指针（推荐）</b></summary>

- 双指针同向遍历每个字符；同时使用哈希表记录每个字符的最新位置；
- 如果右指针遇到已经出现过的字符，则将左指针移动到该字符的位置，更新最大长度；
- 具体细节见代码；

<details><summary><b>Python</b></summary>

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        
        c2p = dict()
        lo = -1  # 左指针
        ret = 1
        for hi, c in enumerate(s):  # 遍历右指针
            if c not in c2p or c2p[c] < lo:  # 如果当前字符还没有出现过，或者出现过但是在左指针的左侧，可以更新最大长度
                ret = max(ret, hi - lo)
            else:  # 否则更新左指针
                lo = c2p[c]

            c2p[c] = hi  # 更新字符最新位置

        return ret
```

</details>


<summary><b>思路2：动态规划</b></summary>

> [最长不含重复字符的子字符串（动态规划 / 双指针 + 哈希表，清晰图解）](https://

**状态定义**
leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/solution/mian-shi-ti-48-zui-chang-bu-han-zhong-fu-zi-fu-d-9/)
- 记 `dp[i] := 以第 i 个字符为结尾的不含重复字符的子串的最大长度`；

**转移方程**
```
dp[i] = dp[i-1] + 1     if dp[i-1] < i-i
      = i-j             else

其中 j 表示字符 s[i] 上一次出现的位置；
```

- 使用一个 hash 表记录每个字符上一次出现的位置；
- 因为当前状态只与上一个状态有关，因此可以使用一个变量代替数组（滚动）；

**初始状态**
- `dp[0] = 1`

<!-- <div align="center"><img src="../_assets/剑指Offer_0048_中等_最长不含重复字符的子字符串.png" height="300" /></div> -->

<details><summary><b>Python</b></summary>

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        idx = dict()
        ret = dp = 0
        for i, c in enumerate(s):
            if c not in idx:
                dp = dp + 1
            else:
                j = idx[c]  # 如果 c 已经出现过，获取其上一个出现的位置
                if dp < i - j:  # 参考双指针思路，这里相当于上一次出现的位置在左指针之前，不影响更新长度
                    dp = dp + 1
                else:  # 反之，在左指针之后
                    dp = i - j

            idx[c] = i  # 更新位置 i
            ret = max(ret, dp)  # 更新最大长度
        return ret
```

</details>

---

### `剑指Offer No.5000 第一个只出现一次的字符 (简单, 2021-12)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![剑指Offer](https://img.shields.io/badge/剑指Offer-lightgray.svg)](题集-剑指Offer.md)

<!--{
    "tags": ["哈希表"],
    "来源": "剑指Offer",
    "编号": "5000",
    "难度": "简单",
    "标题": "第一个只出现一次的字符"
}-->

<summary><b>问题简述</b></summary>

```txt
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。
```

<details><summary><b>详细描述</b></summary>

```txt
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

示例 1:
    输入：s = "abaccdeff"
    输出：'b'
示例 2:
    输入：s = "" 
    输出：' '

限制：
    0 <= s 的长度 <= 50000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->


<summary><b>思路1：哈希表</b></summary>

<details><summary><b>Python</b></summary>

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = defaultdict(int)  # python 3.6 之后 dict 默认就是有序的

        for c in s:
            dic[c] += 1

        for c in s:
            if dic[c] == 1: 
                return c

        return ' '
```

</details>


<summary><b>思路1：有序哈希表</b></summary>

<details><summary><b>Python</b></summary>

- python 3.6 之后 dict 默认就是有序的；
    > [为什么 Python 3.6 以后字典有序并且效率更高？](https://www.cnblogs.com/xieqiankun/p/python_dict.html)

```python
from collections import defaultdict

class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = defaultdict(int)  # python 3.6 之后 dict 默认就是有序的

        for c in s:
            dic[c] += 1

        for c, v in dic.items():
            if v == 1: 
                return c

        return ' '
```

</details>

---

### `程序员面试金典 No.0102 判定是否互为字符重排 (简单, 2022-01)`

[![哈希表](https://img.shields.io/badge/哈希表-lightgray.svg)](技巧-哈希表(Hash).md)
[![程序员面试金典](https://img.shields.io/badge/程序员面试金典-lightgray.svg)](题集-程序员面试金典.md)

<!--{
    "tags": ["哈希表"],
    "来源": "程序员面试金典",
    "编号": "0102",
    "难度": "简单",
    "标题": "判定是否互为字符重排"
}-->

<summary><b>问题简述</b></summary>

```txt
判定给定的两个字符串是否互为字符重排。
```

<details><summary><b>详细描述</b></summary>

```txt
给定两个字符串 s1 和 s2，请编写一个程序，确定其中一个字符串的字符重新排列后，能否变成另一个字符串。

示例 1：
    输入: s1 = "abc", s2 = "bca"
    输出: true 
示例 2：
    输入: s1 = "abc", s2 = "bad"
    输出: false
说明：
    0 <= len(s1) <= 100
    0 <= len(s2) <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/check-permutation-lcci
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：哈希表</b></summary>

<details><summary><b>Python</b></summary>

```python
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2): return False
        
        cnt = [0] * 128
        for c1, c2 in zip(s1, s2):
            cnt[ord(c1)] += 1  # ord 函数用于获取字符的 ascii 码值
            cnt[ord(c2)] -= 1

        return not any(cnt)
```

</details>

<summary><b>思路2：排序</b></summary>

<details><summary><b>Python</b></summary>

```python
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        return sorted(s1) == sorted(s2)
```

</details>

---
