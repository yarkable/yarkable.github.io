---
layout:     post
title:      逆向工具pwntools和pwndbg的心酸安装史
subtitle:   
date:       2019-10-12
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - Reverse Engineering
---

## preface

这两天安装逆向工具 pwntools 和 pwndbg 可把爷给整懵了，由于 IDA Pro 在 Windows 上运行，所以用双系统的话不方便，一般都是虚拟机或者子系统安装这两个工具，但我尝试了各种方法，最后还是在自己双系统 ubuntu 上成功安装，这就来记录一下踩过的坑。

> 最近更新，由于用了 SStap ，我的子系统上也成功安装了， VSCode yes！

## Install pwntools

pwntools 是基于 python2 的一个包，之前听大佬说它也只支持 python2 ，所以一开始我就打算在我的 WSL(windows subsystem for linux) 上安装，然后我就发现 WSL 上面默认就是 python3，并没有安装 python2 ，那也不要紧，就一行命令的事。

```shell
$ sudo apt-get install python python-pip
```

这时我的 WSL 上就有了 python2.7 了，接下去按照大佬的指引，直接 `pip install pwntools` 输入命令行，好像就可以安装了，但是速度十分慢，一直卡着，用了 [pip 豆瓣源](https://www.cnblogs.com/ZhangRuoXu/p/6370107.html)也还是不行，然后我就想直接用 python3 来安装了，我惊奇地发现 WSL 里虽然有 python3 ，但是并没有配对的 pip ，我尼玛傻了都。。

```shell
$ sudo apt-get install python3-pip
```

输入命令之后，系统报错安装 pip3 需要一些依赖，但是这些依赖在我当前源中找不到，google 了一下，才发现是因为我的 apt 的源换成了清华源，把默认的源都注释了，那就去编辑源列表，取消注释就完事了。

```shell
$ sudo vim /etc/apt/sources.list
$ sudo apt-get update
```

哎，我又佛了，这一个 update 搞了我十几分钟还没搞完，估计是最近防火墙太高了，现在连 Google 我也上不去了，重启了 update 无数遍还是没有成功安装 pip3 的依赖，最后就放弃了 WSL 前去双系统了。

我按照自己潜意识里的想法，直接输入了以下命令

```shell
$ pip3 install pwntools
```

嗨他马的现在终于没报错了，看着下载进度条一截一截地增长，我心想终于可以完事了，在经过了几分钟的下载之后，系统提示我成功安装了 pwntools， 我高兴地打开 python 准备试试

```python
from pwn import *
```

下面报了一堆错误，我以为是因为 python 版本的问题，又用 python2 试了一遍还是一样，上网搜索(此时我只能用百度。。) 热心网友告诉我说不能直接用 pip 安装， 要参考作者在 [GitHub 上的教程](https://github.com/Gallopsled/pwntools)安装

```python
apt-get update
apt-get install python3 python3-pip python3-dev git libssl-dev libffi-dev build-essential
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade git+https://github.com/Gallopsled/pwntools.git@dev3
```

好的，那我就照做吧，除了一开始速度有点慢之外整体速度还可以接受，最后又报错，说我没有权限安装，我猛然想起来，**ubuntu 用 pip 安装要用 sudo** ，又用 sudo 权限试了一遍，好像终于可以用 pwntools 了（猛虎落泪）

## Install pwndbg
 
pwndbg 是 linux 系统中的 GDB 的插件，好像没有 pwndbg 的 GDB 十分难用，话不多说，rush！

刚开始也是在 WSL 上安装的 pwndbg，官方给出了[安装教程](https://github.com/pwndbg/pwndbg)

```shell
$ git clone https://github.com/pwndbg/pwndbg
$ cd pwndbg
$ ./setup.sh
```

表面上看是非常简单的，只要把仓库 clone 下来就行了，但是我 WSL 的 git 速度奇慢，下到 15% 就卡在那里不动了，所以我选择了用 windows 的 git 下载，然后在 WSL 里面拷贝进去，这样子做的话理论上是可以的，但我实际做的时候就发现拷贝进去后再执行 `setup.sh` 会出现莫名其妙的错误，说这个脚本有语法错误，最后老老实实在 WSL 中用 git clone 就能直接运行了。

但是过了很久还没有搞完，我就打开脚本看看它里面在干啥，发现他会执行 `apt update` ，emmmm 因为我的源就在国外，所以有时执行这个操作会特别慢，并且我也已经更新过了，所以干脆就把这行给注释了。

![](https://ae01.alicdn.com/kf/H790f811db5f34ce995249cff9af37c0dJ.png)

然后就是漫长的安装中，好在最后总算是安装完成了，在 terminal 中输入 gdb 如果看到了 pwndbg 字样就说明已经成功了

![](https://ae01.alicdn.com/kf/Hd7b7352a04ce43f198c6d61fcabd804bs.png)

同样的方法我在双系统的 ubuntu 中也试了一遍，也成功安装了 pwndbg，等着大佬带着学习吧。

---

这次秃头的安装过程教导我们几个道理
1. 最好看官方的教程，不要轻信网上的文章
2. 在中国做 IT 人员，买个 VPN 非常有必要
3. Google is always your friend
