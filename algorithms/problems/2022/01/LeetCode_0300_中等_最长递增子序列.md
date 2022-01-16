<!-- Tag: 动态规划、贪心 -->

<summary><b>问题简述</b></summary>

```txt
给定整数数组 nums，返回最长严格递增子序列的长度；
进阶：
    你可以设计时间复杂度为 O(N^2) 的解决方案吗？
    你能把时间复杂度降到 O(NlogN) 吗?
```

<details><summary><b>详细描述</b></summary>

```txt
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

示例 1：
    输入：nums = [10,9,2,5,3,7,101,18]
    输出：4
    解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
示例 2：
    输入：nums = [0,1,0,3,2,3]
    输出：4
示例 3：
    输入：nums = [7,7,7,7,7,7,7]
    输出：1

提示：
    1 <= nums.length <= 2500
    -104 <= nums[i] <= 104

进阶：
    你可以设计时间复杂度为 O(n2) 的解决方案吗？
    你能将算法的时间复杂度降低到 O(n log(n)) 吗?

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：动态规划</b></summary>

**状态定义**：`dp[i]` 表示以 `nums[i]` 结尾的最长递增子序列长度；
> 不能将 `dp[i]` 定义 `nums[:i]` 子数组中的最长递增子序列长度，虽然这样定义很直观，但它不满足**最优子结构**的条件，简单来说，就是你无法通过 `dp[i-1]` 得到 `dp[i]`。

<details><summary><b>Python</b></summary>

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        ret = 1
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:  # 如果要求非严格递增，将 '>' 改为 '>=' 即可
                    dp[i] = max(dp[i], dp[j] + 1)
            
            ret = max(ret, dp[i])
        
        return ret
```

</details>


<summary><b>思路2：贪心+二分</b></summary>

- 贪心：如果我们要使上升子序列尽可能的长，则应该让序列上升得尽可能慢，即每次在上升子序列最后加上的那个数尽可能的小。
    > [最长上升子序列 - 力扣官方题解](https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-by-leetcode-soluti/)

<details><summary><b>Python：使用标准库</b></summary>

**写法1**
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        from bisect import bisect_left

        lis = []  # 保存已知的 LIS
        for x in nums:
            idx = bisect_left(lis, x)  # 不能使用 bisect/bisect_right
            if idx == len(lis):  # 插入到末尾
                lis.append(x)
            else:  # 替换
                lis[idx] = x
        return len(lis)
```

**写法2**
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        from bisect import insort_left

        lis = []  # 保存已知的 LIS
        for x in nums:
            idx = bisect_left(lis, x)  # 不能使用 bisect/bisect_right
            if idx == len(lis):  # 插入到末尾
                lis.append(x)
            else:  # 替换
                lis[idx] = x
        return len(lis)
```

</details>

<details><summary><b>Python：不使用标准库</b></summary>

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        def bisect(ls, x):
            l, r = 0, len(ls)
            while l < r:
                m = (l + r) // 2
                if ls[m] <= x:
                    l = m + 1
                else:
                    r = m
            
            return l

        lis = []  #
        for x in nums:
            idx = bisect(lis, x)
            if idx == len(lis):
                lis.append(x)
            else:
                lis[idx] = x
        return len(lis)
```

</details>