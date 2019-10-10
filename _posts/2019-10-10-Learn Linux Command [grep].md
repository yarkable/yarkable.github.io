---
layout:     post
title:      Learn Linux Command [grep]
subtitle:   Grep is your friend
date:       2019-10-10
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
---

今天写一下 Linux 常用的命令之 `grep` 命令，经常用 Linux 的人肯定知道这个命令的强大之处， 俗话说 

> Grep is your friend 



grep 的全称是 `Global Regular Expression Print` ， 顾名思义，将全局正则表达式搜索的内容打印出来。这是个强大的文本搜索工命令，我开始用的时候感觉跟一般编辑器的 `Ctrl+F` 功能差不多，其实它的功能牛逼多了，它可以在一个或者多个文件中搜索字符串模板，或者是正则表达式，然后将匹配到的行输出在屏幕上。一般用于筛选出特定的字符，配合正则表达式使用的话更加灵活。



它的命令格式如下



```	shell
$ grep [option] pattern file1 [file2]...
```



常用的几个 option 有	`-f`	`-n`	`-c`  `-l`	`-i` ，更多的可以去看手册,下面我们用几个例子讲解一下 grep 的一些用法     ，假设我们现在有两个文本文件 file1， file2



```shell
$ cat file1.txt
Fred apples 20
Susy oranges 5
Mark watermellons 12
Robert pears 4
Terry oranges 9
Lisa peaches 7
Susy oranges 12
Mark grapes 39
Anne mangoes 7
Greg pineapples 3
Oliver rockmellons 2  Mellon
Betty limes 14

```



```shell
$ cat file2.txt
Alice peaches
Susan grapes
Jack pineapples
Robert pears
Lucy watermellons
Jennifer apples
Oliver oranges
William mangoes
Bill oranges
```



### 0x01



```shell
$ grep mell file1.txt
Mark watermellons 12
Oliver rockmellons 2  Mellon
```



直接用 grep 在 file1.txt 里面搜索 mell 关键字，可以看到，输出了两行，说明在 file1.txt 中有两行符合匹配规则。



### 0x02



```shell
$ grep mell file1.txt file2.txt
file1.txt:Mark watermellons 12
file1.txt:Oliver rockmellons 2  Mellon
file2.txt:Lucy watermellons
```



这回我们同时在两个文件中搜索，输出时会带上匹配结果所在的文件号。



### 0x03



```shell
$ grep -n mell file1.txt file2.txt
file1.txt:3:Mark watermellons 12
file1.txt:11:Oliver rockmellons 2  Mellon
file2.txt:5:Lucy watermellons
```



`-n` 选项会输出匹配结果所在的行号，方便快速定位。



### 0x04



```shell
$ grep -c mell file1.txt file2.txt
file1.txt:2
file2.txt:1
```



`-c` 选项可以输出模式被匹配的次数， c 就相当于 count。



### 0x05



```shell
$ grep -i MELL file1.txt file2.txt
file1.txt:Mark watermellons 12
file1.txt:Oliver rockmellons 2  Mellon
file2.txt:Lucy watermellons
```



`-i` 选项可以忽略匹配模式的大小写，默认是要区分大小写的。



### 0x06



```shell
$ grep -e mell -e apple file1.txt file2.txt
file1.txt:Fred apples 20
file1.txt:Mark watermellons 12
file1.txt:Greg pineapples 3
file1.txt:Oliver rockmellons 2  Mellon
file2.txt:Jack pineapples
file2.txt:Lucy watermellons
file2.txt:Jennifer apples
```



`-e` 选项可以在一条 grep 语句里面查找多个模式。



### 0x07



现在我们新建了一个文件名叫 `pattern`



```shell
$ cat pattern
mell
apple
```



```shell
$ cat file1.txt | grep -f pattern
Fred apples 20
Mark watermellons 12
Greg pineapples 3
Oliver rockmellons 2  Mellon
```



`-f` 选项可以在文件中读取匹配模式用于匹配。



### 0x08



```shell
$ cat file1.txt file2.txt | grep -E "^L"
Lisa peaches 7
Lucy watermellons
```



`-E` 选项将后面的选项作为一个扩展的正则表达式来用，在本例中就是匹配的 L 开头的行，其实用 `-e` 也能实现相应的效果，但是 `-E` 选项不能匹配多个模式，具体的还是得多用才知道。





> 上面的几个选项有些是可以叠加在一起用的，有时会报错可能是因为顺序不对，例如 grep -nf 可以用，但是 grep -fn 就会报错



更加全面的关于 `grep` 命令的介绍可以看[这篇文章](https://linux.cn/article-5453-1.html)



