<!--{
    "tags": ["字符串"],
    "来源": "+模板",
    "编号": "字符串",
    "难度": "简单",
    "标题": "split切分"
}-->

<summary><b>问题简述</b></summary>

```txt
实现 split(s) 函数：以任意空格切分字符串
示例：
    '  a bc   def   ' -> ['a', 'bc', 'def']
```

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<details><summary><b>Python：简化版</b></summary>

```python
def split(s):
    ret = []
    l, r = 0, 0
    while r < len(s):
        while r < len(s) and s[r] == ' ':  # 跳过空格
            r += 1
        
        l = r  # 单词首位
        while r < len(s) and s[r] != ' ':  # 跳过字符
            r += 1

        if l < r:  # 如果存在字符
            ret.append(s[l: r])
    
    return ret
```

</details>

