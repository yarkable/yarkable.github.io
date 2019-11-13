---
layout:  post
title: 将你的PC变成一个Jupyter服务器
subtitle:   让你随时随地都能愉快写python
date:       2019-07-08
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - python
---

## 前言

前些日子在复习线代，因此避免不了繁杂的行列式计算，关键算出来还不知道对不对，所以想借助 Matlab 来验证。但这玩意太重了，而且复习的时候手头只有一个 iPad，没有 PC，于是我下载了一个叫 Matlab mobile 的 APP，但是一点用也没有，输入输出十分麻烦，而且公网的访问速度也十分感人，于是我想到了可以用局域网搭建一个服务器。

我 PC 端经常用的是 Ubuntu 系统，之前装了 Matlab 后来因为体积太大给卸载了，但是没关系，我电脑中的 Jupyter notebook 配备了几乎全套 python 数据科学包，jupyter notebook 可谓是 python 在数据科学方面的神器了，之前用的时候只需在本地开启一个服务端，然后在浏览器中打开一个客户端和本地服务器建立连接。因此，我们可以选择将这个服务器公开，让局域网中的其他主机也能够访问，而不仅仅在本地访问，
上 google 一搜，还真有办法！

## 原理

我们知道，校园网是一个巨大的局域网，因此我们可以好好利用这个平台，并且局域网的数据传输速度相比公网来说是快得多的，除了 jupyter 服务器，其实还可以用局域网干很多事情，话不多说，直接上教程。

## 安装

关于安装，就不在这里赘述了，直接在命令行输入下面的命令就可以了

 ```python
 pip3 install jupyter-notebook
 ```

如果 python 发行版是 Anaconda 的话是自带 jupyter notebook 的，不需要安装。

安装完成之后在命令行输入以下命令就会新建一个服务器，同时本地的浏览器会打开 jupyter 界面

```bash
$ jupyter-notebook
```

使用教程可以上网搜索，在此不过多讲述。

## 配置

### 配置局域网访问

默认情况下， jupyter notebook 只能在本地访问，要让其在局域网能被访问就得在命令后面加一些参数

```bash
$ jupyter-notebook --ip 0.0.0.0
```

0.0.0.0 就代表公开访问，每次都要输入这么复杂的命令太烦了，可以直接将这一行代码放到一个名为 `jupyter.sh` 的脚本中，以后直接在命令行输入 `./jupyter.sh` 就可以了。

### 配置密码

你可能发现了，jupyter 是用 token 验证身份的，你还得去命令行里找到 token 才能够登录写代码，这不是反人类的操作吗，因此我们需要一个密码，一来方便，而来就算别人登录了你的服务器也不知道密码。

命令行输入以下代码就会在 `.jupyter` 文件夹中生成一个 `jupyter_notebook_config.py` 文件

```bash
$ sudo jupyter notebook --generate-config
```

打开 jupyter，在任意地方输入并执行以下代码即可生成密码

```python
from notebook.auth import passwd
passwd()

```

这是我电脑的画面

![生成密码](https://ae01.alicdn.com/kf/Hf770feb9fa0a40efb9c8bfa1861aee20S.jpg)

生成密码之后再登录的话就会要求输入密码，而不是 token 值，就像下面界面一样。

![密码登录界面](https://ae01.alicdn.com/kf/Hce40433a85374f31a9c581051473278fc.jpg)

## 远程登录

配置完成之后，就可以用其他设备远程登录进行操作了，大体分为以下几个步骤

1. 用 `ifconfig` 命令获取服务器的 ip 地址
2. 客户端(移动设备)连接上校园网 --不需登录
3. 客户端浏览器输入 ip:8888 访问服务器
4. 享受愉快的 python 之旅

---

![jupyter跑神经网络](https://ae01.alicdn.com/kf/H694fb897171449e19e07319ec43916b2C.png)

从此只要连上了学校的 wifi 就可以随时随地访问主机中运行的 Jupyter notebook 了，使用 iPad 配合一个蓝牙键盘写 python 的体验真的很好，尤其是支持自动补全。而且还可以在 jupyter 中新建终端，相当于直接用 ssh 协议访问本地机器，真的很爽，谁用谁知道，而且我还可以在里面跑神经网络，使用跟电脑几乎没差别，在图书馆或寝室想写写代码的话用这个方法再好不过了！



## 更改文件保存路径



默认情况下， jupyter 是将文件保存在用户目录下面的，这样就很恶心，我们可以改变它的保存位置



```
cd 
cd .jupyter
vim jupyter_notebook_config.py
```



打开配置文件之后找到下面这一行，可以看到是被注释的，取消注释，然后在里面填上我们想要保存的位置就行了



```
#c.NotebookApp.notebook_dir = ''
```



