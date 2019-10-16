---
layout:     post
title:      ubuntu嵌入式工具kermit以及oflash使用
subtitle:   嵌入式linux开发工具
date:       2019-06-04
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
---

近期重新开始学习嵌入式，在 ubuntu 下完成对 JZ2440 开发板的配置和开发，kermit 和 oflash 已经有很久没有用了，所以记录一下用法，免得日后又忘了。

## 安装 kermit

```shell
$ sudo apt-get install ckermit
```
注意是ckermit而不是kermit

## 使用方法

### 配置串口
使用前最好在本地用户文件夹下新建一个名为 `.kermrc` 的文件，就是对串口的一些配置，在里面输入以下内容

```
set line /dev/ttyUSB0 
set speed 115200
set carrier-watch off
set handshake none
set flow-control none
robust
set parity none
set stop-bits 1
set file type bin
set file name lit
set rec pack 1000
set send pack 1000
set window 5
```
有时插了串口之后不一定是 /dev/ttyUSB0，可能是 /dev/ttyS0, 并且 serialport 和 openJTAG 一起插的时候会有 /dev/ttyUSB0 和 /dev/ttyUSB1 同时存在，可能会搞错顺序，所以老师叫我们不要同时插入这两个

### 连接串口

插入串口的情况下，在终端输入以下命令就可以进入软件界面
```shell
$ sudo kermit
```
![](https://ae01.alicdn.com/kf/HTB1MrUVboKF3KVjSZFEq6xExFXaK.jpg)

此时，输入 `connect` 便可以进入命令界面与本机的串口进行通信，前提是有串口，以下是我在 Nor flash 下用 uboot 输出的信息，说明已经成功连接到了开发板

![](https://ae01.alicdn.com/kf/HTB1JnwVbgKG3KVjSZFLq6yMvXXao.jpg)

也可以用以下命令直接进入命令模式
```shell
$ sudo kermit -c
```

### 断开串口
断开和串口的连接进入 ubuntu 环境时可以用 `Ctrl + \` 再加  `C`，这个在工具中也有说到，想要再次连接时直接输入 `connect` 即可

---

## oflash 使用

这是将二进制文件烧写到 JZ2440 要用到的工具，使用方法也很简单，大致是

```shell
$ sudo ./path/oflash 0 1 0 0 0 /path/xx.bin
```

先找到 oflash 所在位置，运行，然后输入 0 1 0 0 0,这是一些配置，包括用的是 Nor flash 还是 Nand flash，在哪片内存进行烧写，是 S3C2440 还是 S3C2410 等等，不写的话就要在命令行一个一个输入，有点繁琐，最后一个参数就是准备烧写的二进制文件

![](https://ae01.alicdn.com/kf/HTB12SgZbliE3KVjSZFMq6zQhVXaP.jpg)

等待他出现 Epppp 字样就烧写完毕了，注意烧写的过程中要给板子上电