---
layout:     post
title:      ubuntu配置ssr注意事项
subtitle:   真的希望人人都能用上 Google
date:       2019-10-18
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
---



## preface



国庆过去不久，国内抓 VPN 也放松了些，然后趁着我的服务器还没过期，就在我的 Google-cloud 上重新搭建了个 ssr 环境，发现 ubuntu 上配置这些还真是没有 windows 那么轻松，干脆整个记录一下，以供新入坑 ubuntu 的同志们康康。



## 拥有 VPS



首先，你得拥有一个国外的 VPS(Virtual private server)，也就是要有一台在大陆外面的虚拟主机，这样才不会被墙，这个可以自己去买，我之前用的是别人配置好的节点，然后现在也有一台在台湾的 Google-cloud 云服务器，只要有 VISA 卡的都可以免费撸一年(滑稽)，关于 Google-cloud 搭建 ssr 服务器的可以看[这篇教程](https://www.wmsoho.com/google-cloud-platform-ssr-bbr-tutorial/)



## 安装客户端



假设你现在已经配置好了你的节点，那就要在本机上安装 ssr 客户端了，用下面这些命令就能够安装小飞机，是个基于 QT 开发的图形界面



```shell
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```



![ss_logo](https://ae01.alicdn.com/kf/Hd1eb9e3b65f54986b1dfd56068d15a2cE.png)



## 配置节点



ubuntu 的这个软件很讨厌，不能添加订阅，只能一个服务器一个服务器地添加，像我买的节点上有十多个服务器，如果要我自己手动配置的话就很麻烦，所以我一般只添加一个。打开 `连接 -> 添加 -> 手动` 就可以看到下面的配置表了，具体配置表如下：



![config](https://ae01.alicdn.com/kf/H1763b30c8b4e4d3bbee63ecdadd8b863j.png)



`配置名称`自己设定，`服务器名称、端口、密钥、加密方式`依个人的服务器而定，如果是买的节点，通常都会告诉你这些信息，自己搭建的话可以自定义这些数据，本地的那三个就跟我上面的配置一样就行了，勾选上最后两个选项：自动化会在程序启动时自己连接服务器，调试功能可以把操作记录写入日志中，最后，点击 `OK` 。



## 修改本地代理



上述操作完成之后，我们还需要进行一步，否则我们还是上不了 Google ，打开 `系统设置 -> 网络 -> 网络代理` ，看到如下界面，刚刚我们的服务器设置了本地端口为 `1080` ，所以我们把这里所有的代理全都改成 `127.0.0.1:1080` ，方法设置为手动，点击 `应用到整个系统`， 输入 root 密码完事。



![proxy](https://ae01.alicdn.com/kf/H7b0bc550eaba410bb835c3335b255cbfH.png)



打开 chrome 浏览器，不出意外的话，输入 `www.google.com` 是可以看到这个界面的，那就大功告成了，开始享受 Google 带来的便利。



![google](https://ae01.alicdn.com/kf/Hdcdeabe85c594affa9eb916e8272d5bc3.png)



## 后续 



前面不是说了吗，这个玩意一次只能添加一个服务器，今天我在我 Google-cloud 上搭建了个节点准备测试一下速度来着，结果按照上面的配置添加节点后竟然说无法启动这个服务器？？？搜索了一下，说是因为我的 `1080` 端口被占用了，按照下面的命令打了一遍发现是小飞机那个客户端占用了 `1080` 端口，然后杀掉这个进程，再重新开启，还是一样无法启动。



最后我就把自己的节点的本地端口变成 `1090` ，让 Google-cloud 使用 `1080` 端口，这样就能够成功使用 Google-cloud 来上外网了，试过了，网速挺不错的，甚至还可以看 YouTube  `:P`



![port](https://ae01.alicdn.com/kf/H31f4a8c05aa64a74b1953bc42b31be0fn.png)