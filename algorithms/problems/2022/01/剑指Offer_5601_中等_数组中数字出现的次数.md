<!--{
    "tags": ["位运算"],
    "来源": "剑指Offer",
    "编号": "5601",
    "难度": "中等",
    "标题": "数组中数字出现的次数"
}-->

<summary><b>问题简述</b></summary>

```txt
一个整型数组中除两个数字外，其他数字都出现了两次。求这两个只出现一次的数字。
要求时间复杂度是O(n)，空间复杂度是O(1)。
```

<details><summary><b>详细描述</b></summary>

```txt
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

示例 1：
    输入：nums = [4,1,4,6]
    输出：[1,6] 或 [6,1]
示例 2：
    输入：nums = [1,2,10,4,1,4,3,3]
    输出：[2,10] 或 [10,2]

限制：
    2 <= nums.length <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 异或运算的性质：
    ```
    性质1：0^a = a
    性质2：a^a = 0
    性质3（交换律）：a^b = b^a
    性质4（结合律）：(a^b)^c = a^(b^c)
    ```
- 根据性质1 和性质2，可以构造如下算法：
    ```
    定义 all_xor(nums) := nums[0] ^ nums[1] ^ .. ^ nums[-1]
    记这两个不同的数分别为 a 和 b
    则 ab = a ^ b = all_xor(nums)  # 存在两个相同数字的都被消去
    因为 a != b，则 ab 的二进制表示中必然有一个为 1（因为 0^1=1）
    根据这个位置的 1 将 nums 分为两组 ls 和 rs
    则结果为 [all_xor(ls), all_xor(rs)]
    ```


<details><summary><b>Python</b></summary>

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        
        def all_xor(ns):
            r = 0
            for x in ns:
                r ^= x
            return r

        # 求 a^b
        ab = all_xor(nums)

        # 找出二进制 ab 中从左往右第一个 1 的索引
        idx = 0
        while not ab & 1:
            ab >>= 1
            idx += 1
        
        # 将 nums 根据 1 的位置分为两部分
        y = 1 << idx
        ls, rs = [], []
        for x in nums:
            if x & y:
                ls.append(x)
            else:
                rs.append(x)
        
        return [all_xor(ls), all_xor(rs)]
        
```

</details>

