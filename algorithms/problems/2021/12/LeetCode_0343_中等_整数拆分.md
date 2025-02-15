<!--{
    "tags": ["数学", "动态规划"],
    "来源": "LeetCode",
    "编号": "0343",
    "难度": "中等",
    "标题": "整数拆分"
}-->

<summary><b>问题简述</b></summary>

```txt
给定一个正整数 n，将其拆分为至少两个正整数的和，使这些整数的乘积最大化。返回最大乘积。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:
    输入: 2
    输出: 1
    解释: 2 = 1 + 1, 1 × 1 = 1。
示例 2:
    输入: 10
    输出: 36
    解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
说明: 你可以假设 n 不小于 2 且不大于 58。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/integer-break
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：动态规划</b></summary>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

- 在不使用任何数学结论的前提下，可以把本题当做纯 DP 来做：

<details><summary><b>Python（写法1）</b></summary>

> LeetCode 官方题解中的写法：[整数拆分](https://leetcode-cn.com/problems/integer-break/solution/zheng-shu-chai-fen-by-leetcode-solution/)

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [1] * (n + 1)

        for i in range(2, n + 1):
            for j in range(1, i):
                # 状态定义：dp[i] 表示长度为 i 并拆分成至少两个正整数后的最大乘积（i>=1）
                #   j * (i - j)   表示将 i 拆分成 j 和 i-j，且 i-j 不再拆分
                #   j * dp[i - j] 表示将 i 拆分成 j 和 i-j，且 i-j 会继续拆分，dp[i-j] 即为继续拆分的最优结果（最优子结构）
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))

        return dp[n]
```

</details>

<details><summary><b>Python（写法2）</b></summary>

> 《剑指Offer》中的写法

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        # 对于 n = 2、3 的情况，直接硬编码
        if n == 2:
            return 1
        if n == 3:
            return 2

        # 状态定义：dp[i] 表示长度为 i 并拆分成至少两个正整数后的最大乘积（i>3）
        #   当 i <= 3 时，不满足该定义，此时不拆效率最高
        #   初始状态（dp[0] 仅用于占位）
        dp = [0,1,2,3] + [0] * (n - 3) 

        for i in range(4, n + 1):
            for j in range(2, i):
                dp[i] = max(dp[i], dp[i-j] * dp[j])

        return dp[n]
```

</details>


<summary><b>思路2：数学/贪心</b></summary>

- 数学上可证：尽可能按长度为 3 切，如果剩余 4，则按 2、2 切；
  > 证明见：[剪绳子1（数学推导 / 贪心思想，清晰图解）](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/solution/mian-shi-ti-14-i-jian-sheng-zi-tan-xin-si-xiang-by/)

- **简述**：当 `x >= 4` 时，有 `2(x-2) = 2x - 4 >= x`；简言之，对任意大于等于 4 的因子，都可以拆成 2 和 x-2 而不损失性能；因此只需考虑拆成 2 或 3 两种情况（1除外）；而由于 `2*2 > 3*1` 和 `3*3 > 2*2*2`，可知最多使用两个 2；

<details><summary><b>Python</b></summary>

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        import math
        if n <= 3:
            return n - 1
        
        a, b = n // 3, n % 3
        if b == 1:
            return int(math.pow(3, a - 1) * 4)
        elif b == 2:
            return int(math.pow(3, a) * 2)
        else:
            return int(math.pow(3, a))
```

</details>

