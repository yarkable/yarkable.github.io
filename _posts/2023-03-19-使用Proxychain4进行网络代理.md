---
layout:     post
title:      使用Proxychain4进行网络代理
subtitle:   
date:       2023-03-19
author:     li hao
header-img: img/green-bg.jpg
catalog: true
tags:
    - ssh
    - linux
    - tools

---

## 背景



学校的个人账号只能在一台设备上进行认证联网，但是我们使用的 GPU 服务器经常需要访问互联网，在服务器上认证之后我们自己的电脑就会掉线，所以可以通过代理的方式让服务器通过我们自己的设备进行联网，解决这个问题。

> 本文在实验室师弟写的 pdf 版本教程上改编而来，方便自己查阅



## 安装软件



1. 主要是通过 proxychains-ng 来转发网络请求，可以通过 git 下载也可以直接下载压缩包。

```bash
git clone https://github.com/rofl0r/proxychains-ng
```

2. 然后进入软件目录，用 `pwd` 命令看一下当前的绝对路径，这个在下一步中要用到
3. 进入目录执行命令，这里的 pwd 就是上一步输出的绝对路径，**要输绝对路径**，不然后面编译的时候会出错

```bash
./configure --prefix=pwd --sysconfdir=pwd
```

4. 安装二进制文件（make install-config 之后会生成一个配置文件 proxychains.conf）

```bash
make -j
make install
make install-config
```



## 配置



进入安装目录找到配置文件 proxychains.conf，进行编辑，在底部添加需要代理的设备的 ip 和端口，我使用的 clash，是 socks 代理，所以我的配置是

```bash
socks5 172.31.xx.xx 7879
```

那么我们自己的设备上也需要打开代理软件才能让服务器访问到网络，在 clash 中打开 `Allow LAN`， v2ray 中打开 `允许局域网的连接` 就行了。这样我们的设备可以访问的东西，服务器都可以访问到。



此外，我们还需要在 bash 配置文件中加入二进制文件的路径，不然运行时会找不到文件（如果是通过管理员装的，则不用这一步）

```bash
vim ~/.bashrc
export PATH=/data/xxx/proxychains/bin:$PATH 
export PROXYCHAINS_CONF_FILE=/data/xxx/proxychains/proxychains.conf
```

完事以后重新打开一个终端就生效了，`source ~/.bashrc` 我试过是没有效果的，建议直接新开一个终端使用。

## 使用



在想要代理网络的时候就在命令前加上 `proxychains4` 就可以了，例如

```bash
proxychains4 curl cip.cc 
proxychains4 python main.py
```

## troubleshoot



在使用的时候报错找不到 proxychains.conf 的，基本上都是编译的时候没有填绝对路径而是填了相对路径，用 `make uninstall` 以及 `make clean` 把刚刚生成的东西给删了，然后重新运行上述的安装步骤，一定要填绝对路径。