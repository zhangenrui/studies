#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time: 2022-02-10 7:35 下午

Author: huayang

Subject:

"""
from typing import *


class Solution1:
    def knapsack(self, V: int, n: int, vw: List[List[int]]) -> int:
        # write code here

        dp = [[0] * (V + 1) for _ in range(n + 1)]
        # 对应递归基：剩余容量为 V 时前 0 个物品的最大重量
        dp[0][V] = 0

        for i in range(1, n + 1):
            for rest in range(V + 1):  # 这里正序逆序遍历都可以
                # 与 dfs 的过程一一对应
                r1 = dp[i - 1][rest]
                r2 = 0
                if rest >= vw[i - 1][0]:
                    r2 = dp[i - 1][rest - vw[i - 1][0]] + vw[i - 1][1]
                dp[i][rest] = max(r1, r2)

        return dp[n][V]


class Solution2:
    def knapsack(self, V: int, n: int, vw: List[List[int]]) -> int:
        # write code here

        dp = [0] * (V + 1)
        dp[0] = 0  # 可以省略

        for i in range(1, n + 1):
            for rest in range(V, vw[i - 1][0] - 1, -1):
                # 不拿第 i 个物品
                r1 = dp[rest]
                # 拿第 i 个物品
                r2 = dp[rest - vw[i - 1][0]] + vw[i - 1][1]
                # 取较大的
                dp[rest] = max(r1, r2)

        return dp[V]


class Solution3:
    def knapsack(self, V: int, n: int, vw: List[List[int]]) -> int:
        # write code here

        # 最大重量
        W = sum(it[1] for it in vw)

        # 初始化为无穷大
        dp = [[float('inf')] * (W + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # 重量为 0 所需的最小空间也是 0

        for i in range(1, n + 1):
            for w in range(W + 1):
                r1 = dp[i - 1][w]
                r2 = float('inf')
                if w - vw[i - 1][1] >= 0:
                    r2 = dp[i - 1][w - vw[i - 1][1]] + vw[i - 1][0]
                dp[i][w] = min(r1, r2)

        for w in range(W, -1, -1):
            if dp[n][w] <= V:
                return w

        return 0


class Solution4:
    def knapsack(self, V: int, n: int, vw: List[List[int]]) -> int:
        # write code here

        # 最大重量
        W = sum(it[1] for it in vw)

        # 初始化为无穷大
        dp = [float('inf')] * (W + 1)
        dp[0] = 0  # 重量为 0 所需的最小空间也是 0

        for i in range(1, n + 1):
            for w in range(W, vw[i - 1][1] - 1, -1):
                dp[w] = min(dp[w], dp[w - vw[i - 1][1]] + vw[i - 1][0])

        # 逆序遍历 S，当找到需要的最小体积相遇等于 V 时，此时的 w 就是最大重量
        for w in range(W, -1, -1):
            if dp[w] <= V:
                return w

        return 0


def random_input():
    import random
    MAX = 1000

    V = random.randint(1, MAX)
    n = random.randint(1, 100)  # 因为 方法 3, 4 比较慢，所以控制一下 n 的范围

    vw = []
    for _ in range(n):
        v, w = random.randint(1, MAX), random.randint(1, MAX)
        vw.append([v, w])

    return V, n, vw


def _test():
    """"""
    for _ in range(10):
        V, n, vw = random_input()
        r1 = Solution1().knapsack(V, n, vw)
        r2 = Solution2().knapsack(V, n, vw)
        r3 = Solution3().knapsack(V, n, vw)
        r4 = Solution4().knapsack(V, n, vw)

        assert r1 == r2 == r3 == r4


if __name__ == '__main__':
    """"""
    _test()
