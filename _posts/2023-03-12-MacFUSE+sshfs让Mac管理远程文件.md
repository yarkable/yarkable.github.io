---
layout:     post
title:      MacFUSE+sshfs让Mac管理远程文件
subtitle:   
date:       2023-03-12
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - ssh
    - linux
    - macOS
    - tools

---

## 背景



在 MacBook 上开发变多，经常需要查看远程服务器上的图片，因此挂载远程目录是最方便的做法。Windows 上有 sshfs manager 这么优秀的 GUI 方便操作，但是 macOS 系统没有类似的替代品（要么就是太古老用不了），所以得用命令行手动挂载，这里记录一下。



## 安装软件



要映射远程文件夹，需要两个工具：MacFUSE 和 sshfs。这两个文件都可以从 [osxfuse 网站]( https://osxfuse.github.io/)上下载，傻瓜式安装就行。先安装完 MacFUSE，在设置界面拉到最下面会有 MacFUSE 的 logo，然后重启电脑，安装 sshfs。



## 映射文件夹



跟 windows 一样的，但是这里要加上 `-ovolname` 参数，方便辨认，不然的话默认的名字是 `macFUSE Volume 0 (sshfs)`，很难记，而且映射多个文件夹时很难记住映射对应的文件夹。

```shell
sshfs <用户名>@<服务器>:<服务器上的绝对路径> <本地目标文件夹> -ovolname=<映射后的文件夹名称>
```

取消映射的话可以在 finder 里面右键点击对应的盘符然后`推出xxx文件夹`，也可以在命令行里直接 `umount 文件夹`。挂载的文件夹很多的话可以写个脚本方便管理。



## reference



https://xmanyou.com/mac-mount-remote-folder/