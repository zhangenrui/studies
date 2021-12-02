常见报错
===

## 文件编码
> 报错：`UnicodeDecodeError: 'utf-8' codec can't decode byte xxx ...`

1. 使用相关编辑器确认文件的真实编码；
2. 如果编码正确还报错，那可能是人为的插入了一些特殊符号，破坏了部分字符，此时可以在打开文件时，添加 `errors='ignore'` 参数，忽略这些错误；
   ```python
   f = open(fp, encoding='utf8', errors='ignore')
   # 注意，指定 errors='ignore' 可能会导致文件数据丢失
   ```
