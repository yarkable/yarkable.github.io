---
layout: post
title: 将Linux服务器目录映射到Windows的方法
subtitle: 
date: 2020-07-15
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - object detection
    - deep learning
---



## preface



本人日常主用 Windows 系统，然后 ssh 到服务器上进行 code，所以涉及文件传输都用 scp 命令或者直接用 mobaXTerm 进行，还是有些小不方便的，因为我还是得将东西下载到自己的 Windows 上再传到 Linux 服务器上。上次看师兄在群里分享了一个软件可以直接将服务器上的目录挂在到 Windows 的资源管理器，相当于多了一个磁盘，这样子就可以直接将数据下载到服务器上了，省去了一步操作，挺方便的，这就记录一下。



## installation



这玩意叫 SSHFS-Win ，[GitHub 官方仓库](https://github.com/billziss-gh/sshfs-win)都有指示页，其实就下载两个文件就行了，照着安装起来



![install](https://i.loli.net/2020/07/19/Ny24MhnQVKOtDlA.png)



## use



安装完的话就可以用了，Windows 下打开资源管理器，右击 `此电脑` ，选择 `映射网络驱动器`

![mapping](https://i.loli.net/2020/07/19/FfsJxvZbR5mKnoC.png)



然后文件夹中填上前缀 `\\sshfs\`，后面就是自己服务器的用户名和 ip 地址，和登录 ssh 服务器是一样操作的。还可以直接加上自己服务器上的文件夹，很方便。

![图片.png](https://i.loli.net/2020/07/19/ClqsQ5A9fcNDznY.png)



上一步填完信息之后，会让我们填入服务器相对应的密码，我这里就不展示了，成功了之后呢，在资源管理器上面就会出现一个新的卷，将服务器上的目录映射到了本地，就可以像操作自己电脑文件一样操作远程服务器了。



![server](https://i.loli.net/2020/07/19/cts1gTz348KfSaD.png)