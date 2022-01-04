<!-- Tag: 动态规划、经典 -->

<summary><b>问题简述</b></summary>

```txt
我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。
求按从小到大的顺序的第 n 个丑数。
```

<details><summary><b>详细描述</b></summary>

```txt
我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

示例:
    输入: n = 10
    输出: 12
    解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
说明:
    1 是丑数。
    n 不超过1690。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/chou-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：动态规划</b></summary>

> [丑数（动态规划，清晰图解）](https://leetcode-cn.com/problems/chou-shu-lcof/solution/mian-shi-ti-49-chou-shu-dong-tai-gui-hua-qing-xi-t/)

<details><summary><b>Python</b></summary>

- 关于这份代码的理解，可以参考：[丑数，清晰的推导思路](https://leetcode-cn.com/problems/chou-shu-lcof/solution/chou-shu-ii-qing-xi-de-tui-dao-si-lu-by-mrsate/)

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:

        dp = [1] * n
        a, b, c = 0, 0, 0

        for i in range(1, n):
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(n2, n3, n5)

            if dp[i] == n2: 
                a += 1
            if dp[i] == n3: 
                b += 1
            if dp[i] == n5: 
                c += 1
        
        return dp[-1]

```

</details>
