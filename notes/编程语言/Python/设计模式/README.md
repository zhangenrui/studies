Python 设计模式
===

- [单例模式](#单例模式)


## 单例模式
- 记录一种最简单的实现方式；
- Python 的模块就是天然的单例模式；因此可以使用如下方式实现单例

    ```python 
    class _Some:
        """"""
        cnt = 0

    some = _Some()  # 从其他模块导入 some 即可
    ```

- 更多实现方式请参考：[Python中的单例模式的几种实现方式的及优化 - 听风。](https://www.cnblogs.com/huchong/p/8244279.html)