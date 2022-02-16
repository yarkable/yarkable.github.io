---
layout: post
title: python调试工具ipdb基本命令
subtitle: 
date: 2022-01-03
author: xmfbit
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - python
---



## preface 



这玩意就是 python 内置调试工具 pdb 的升级版，拥有代码高亮等功能，方便人性化使用，其实命令跟 gdb 是差不多的，但是每次用都要去网上查有点麻烦，找到一篇不错的文章，基本的命令都有了，以后直接看就行了

> 转载于：[使用IPDB调试Python代码 | 来呀，快活呀~ (xmfbit.github.io)](https://xmfbit.github.io/2017/08/21/debugging-with-ipdb/)



## 安装与使用



IPDB 以 Python 第三方库的形式给出，使用 `pip install ipdb` 即可轻松安装。

在使用时，有两种常见方式。



### 集成到源代码中



通过在代码开头导入包，可以直接在代码指定位置插入断点。如下所示：

```
import ipdb
# some code
x = 10
ipdb.set_trace()
y = 20
# other code
```



则程序会在执行完 `x = 10` 这条语句之后停止，展开 Ipython 环境，就可以自由地调试了。



### 命令式



上面的方法很方便，但是也有不灵活的缺点。对于一段比较棘手的代码，我们可能需要按步执行，边运行边跟踪代码流并进行调试，这时候使用交互式的命令式调试方法更加有效。启动IPDB调试环境的方法也很简单：

```
python -m ipdb your_code.py
```



## 常用命令



IPDB调试环境提供的常见命令有：



### 帮助



帮助文档就是这样一个东西：当你写的时候觉得这TM也要写？当你看别人的东西的时候觉得这TM都没写？

使用 `h` 即可调出 IPDB 的帮助。可以使用 `help command` 的方法查询特定命令的具体用法。



### 下一条语句



使用 `n`(next) 执行下一条语句。注意一个函数调用也是一个语句。如何能够实现类似 “进入函数内部” 的功能呢？



### 进入函数内部



使用 `s`(step into) 进入函数调用的内部。



### 打断点



使用 `b line_number`(break) 的方式给指定的行号位置加上断点。使用 `b file_name:line_number` 的方法给指定的文件（还没执行到的代码可能在外部文件中）中指定行号位置打上断点。

另外，打断点还支持指定条件下进入，可以查询帮助文档。



### 一直执行直到遇到下一个断点



使用 `c`(continue) 执行代码直到遇到某个断点或程序执行完毕。



### 一直执行直到返回



使用 `r`(return) 执行代码直到当前所在的这个函数返回。



### 跳过某段代码



使用 `j line_number`(jump) 可以跳过某段代码，直接执行指定行号所在的代码。



### 更多上下文



在IPDB调试环境中，默认只显示当前执行的代码行，以及其上下各一行的代码。如果想要看到更多的上下文代码，可以使用 `l first[, second]`(list) 命令。

其中 `first` 指示向上最多显示的行号，`second` 指示向下最多显示的行号（可以省略）。当 `second `小于 `first` 时，`second`指的是从 `first` 开始的向下的行数（相对值vs绝对值）。

根据 [SO上的这个问题](https://stackoverflow.com/questions/6240887/how-can-i-make-ipdb-show-more-lines-of-context-while-debugging)，你还可以修改IPDB的源码，一劳永逸地改变上下文的行数。



### 我在哪里



调试兴起，可能你会忘了自己目前所在的行号。例如在打印了若干变量值后，屏幕完全被这些值占据。使用 `w` 或者 `where` 可以打印出目前所在的行号位置以及上下文信息。



### 这是啥



我们可以使用 `whatis variable_name` 的方法，查看变量的类别（感觉有点鸡肋，用 `type` 也可以办到）。



### 列出当前函数的全部参数



当你身处一个函数内部的时候，可以使用 `a`(argument) 打印出传入函数的所有参数的值。



### 打印



使用 `p`(print) 和 `pp`(pretty print) 可以打印表达式的值。



### 清除断点



使用 `cl` 或者 `clear file:line_number` 清除断点。如果没有参数，则清除所有断点。



### 再来一次



使用 `restart` 重新启动调试器，断点等信息都会保留。`restart `实际是 `run` 的别名，使用 `run args` 的方式传入参数。



### 退出



使用 `q` 退出调试，并清除所有信息。