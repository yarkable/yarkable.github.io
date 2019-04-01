---
layout:     post
title:      解决Ubuntu下MatLab许可证错误
subtitle:   这还真是个磨人的小妖精
date:       2019-04-01
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - MatLab
---

之前在 Ubuntu 下经常要相机标定，所以下载了 MatLab，因为一个 license manger error 搞了我大半天才弄好，然后最近实验室的小伙伴也在 Ubuntu 装了个 MatLab，也是遇到了这个问题，两天都没装好，最后终于找到了解决方法，于是记录下来，以防下次重装还遇到这个错误。

我用的是学校校园网下载的 Mac/Win/Liunx 三合一版本 MatLab，windows 方法也相同，反正道理都是一样的。先下载解压之后，用下列命令给整个文件夹都赋予权限

```bash
$ sudo chmod 777 matlab_xxx
```

然后执行里面的 install 脚本，注意要以超级管理员的身份执行，不然之后安装的时候会提示无法安装至 /usr/local/matlab_xx 里面

```bash
$ sudo ./install.sh
```

之后会问你是否要在 /usr/local/bin 中建立一个链接，选择**是**，接下来就按部就班，输入账号后就等着安装进度条变满，就安装好了。

但是在命令行里输入 matlab 的时候，就提示 license manger error -9，好像是什么用户名不匹配，就很烦，具体截图我也没了，所以只是说一下做法。

1. 去 MathWork 网站上登陆你的账号，然后为你的 Linux 主机添加许可证，会让你填主机 ID 和用户名，用户名就是你登陆 Linux 的名字，主机 ID 是你的 MAC 地址，用`ifconfig`可以查看。
2. 将生成的许可证下载到电脑本地（不下载不知道行不行，没试过）。
3. 进入 MatLab 目录，不是进入安装包！我就是在这里被坑了，安装目录的地址在 `/usr/local/matlab/matlab20xx`，里面会找到一个`activate.sh`，直接执行这个脚本，将刚刚下载到本地的许可证添加进去或者用 MathWork 账号登陆就会显示激活成功

```bash
$ ./activate.sh
```

通过以上步骤再从命令行输入 matlab 时应该就能成功开启了，但是只能用命令行，从 Dash 界面点击图标是不会有反应的，这个我也不知道为什么。
