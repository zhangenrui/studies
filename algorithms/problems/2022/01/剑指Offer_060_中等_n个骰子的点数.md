<!-- Tag: 动态规划、数学、概率、经典 -->

<summary><b>问题简述</b></summary>

```txt
把 n 个骰子扔在地上，所有骰子朝上一面的点数之和为 s。
输入 n，打印出 s 的所有可能的值出现的概率（按 s 从小到大排列）。
```

<details><summary><b>详细描述</b></summary>

```txt
把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

示例 1:
    输入: 1
    输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
示例 2:
    输入: 2
    输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]

限制：
    1 <= n <= 11

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 参考“跳台阶”：`dp[s] = dp[s-1] + dp[s-2] + .. + dp[s-6]`；
    - 具体来说，对 `n` 个骰子，有 `5*n+1` 种可能，每种可能当做是要跳的台阶数；
    - 假设目标台阶数是 `j`，上一轮能达到的台阶序列为 `dp`；
    - 则这一轮下，`dp_new[j] = dp[j-1] + dp[j-2] + .. + dp[j-6]`（因为每一轮的范围是不一致的，所以新起一个 dp）；
        > 也可以只是用一个 dp，参考：[n个骰子的点数 - 路漫漫我不畏](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/nge-tou-zi-de-dian-shu-dong-tai-gui-hua-ji-qi-yo-3/)
    - 注意范围，比如当 `n=2, j=2` 时，上一轮掷出的 2~6 就都是无效的，即保证 `dp[j - k]` 合法，其中 `k` 为当前骰子掷出的点数；


<details><summary><b>Python</b></summary>

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        
        dp = [0] + [1] * 6  # dp[0] 不使用

        for i in range(2, n + 1):
            dp_new = [0] * (6 * i + 1)  # 浪费一点空间，便于理解
            for j in range(i, 6 * i + 1):
                for k in range(1, 7):
                    if 0 < j - k <= (i-1) * 6:
                        dp_new[j] += dp[j - k]
            dp = dp_new
        
        return [i / 6**n for i in dp[n: 6 * n + 1]]
```

</details>


**优化**：为了区分，上述思路记为“逆向递推”，它需要额外的判断来确保合法，下面的“正向递推”可以省略这个判断；
> [n 个骰子的点数（动态规划，清晰图解） - Krahets](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/jian-zhi-offer-60-n-ge-tou-zi-de-dian-sh-z36d/)

<details><summary><b>Python</b></summary>

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        
        # 只有一个骰子时，各数字都只有一种可能
        dp = [1] * 6

        for i in range(2, n + 1):
            dp_new = [0] * (5 * i + 1)  # i 个骰子，s 共有 5 * i + 1 种可能，初始化为 0
            for k in range(6):  # 当前骰子掷出 6 个可能之一
                for j in range(len(dp)):  # 前 i - 1 个骰子的可能序列
                    dp_new[k + j] += dp[j]  # 这里有点像“跳台阶”，即 dp[s] = dp[s-1] + dp[s-2] + .. + dp[s-6]
            dp = dp_new
        
        return [i / 6**n for i in dp]
```

</details>

