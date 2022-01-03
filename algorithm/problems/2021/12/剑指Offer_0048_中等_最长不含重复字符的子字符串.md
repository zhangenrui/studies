<!-- Tag: 双指针 -->

<summary><b>问题简述</b></summary>

```txt
从字符串中找出最长的不包含重复字符的子字符串，返回其长度。
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

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：双指针</b></summary>

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

