---
layout:     post
title:      用Shell脚本实时监测进程
subtitle:   妈妈再也不用担心我的程序跑飞了
date:       2019-03-30
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - shell
---

## 前言

最近做项目有需要将程序设置为自启动，所以弄了个类似看门狗的程序检测脚本。

之前在网上复制的脚本根本就是假的，害我检查了好久，最后还是自己修修改改才成功了，下面就是整个脚本文件，只需要把名字和路径替换一下就行了。

```bash
#!/bin/bash
sec=6
name=Your_thread_name
Thread=`ps -ef | grep $name | grep -v "grep"`
while [ 1 ]
do
count=`ps -ef | grep $name | grep -v "grep" | wc -l`
echo "Thread count: $count"
    if [ $count -gt 0 ]; then
        echo "The thread is still alive!"
        sleep $sec
    else 
        echo "Starting process..."
        cd Your_directory/
        gnome-terminal -x bash -c "./$name;exec bash;"
        sleep 2
        echo "Poocess has started!"	
    fi
done

```

## 原理

原理就是不断在后台检测你的程序有没有在跑，在跑的话就过几秒再次检测，没在跑的话(跑飞了)就马上运行程序，不得不说 shell 还是非常强大的。

其中`gnome-terminal -x bash -c "./$name;exec bash;"`这句话非常棒，是我在网上搜到的一个命令，新开一个命令行执行你的脚本，不跟看门狗冲突，让我们可以实时看到看门狗的输出。

## BUG

之前我用网上复制的脚本，一直出错，显示 `xx command not find`，我认真对比了好几遍额都没发现哪里错了，而且把这个命令在命令行里单独敲出来也没有错。听网友说用 vim 打开脚本，输入 `:set ff`可能是 dos 风格的文件，改成 unix 就行了，然后我按照教程查看，这脚本本身就是 unix 分风格的呀，整得我一脸懵逼。

![vim图](https://ae01.alicdn.com/kf/HTB1NDtpOOrpK1RjSZFh760SdXXaS.png)

然后我就新建一个脚本文件，一行一行重新手敲，敲一行就运行一次， 完完全全一模一样的两个文件，自己手打的就没有问题，网上复制的就不行，行吧！我服了。

在网上搜索的时候也有人遇到过我一样的问题，但是没有人解决，重新敲一遍就好了，具体原理我也不知道，玄学。
