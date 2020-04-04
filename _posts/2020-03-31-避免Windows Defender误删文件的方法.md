---
layout: post
title: 避免Windows Defender误删文件的方法
subtitle: 这玩意戏太多了吧
date: 2020-03-31
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - troubleshoot
---



## preface



作为一个有点网络安全知识的菜鸡，经常会用到一些渗透工具和脚本，有些时候，刚下载完的工具就会直接被 Windows Defender(下面简称 WD) 给删除，真的心态爆炸，拜托，爷就是搞安全的，我自己写的脚本有没有病毒我不比你有数嘛？本来把 WD 都已经关掉了，下载的工具不会被删，最近写个木马脚本又提示说有危险，写好保存之后直接给我删了，wtm，想干死他



![defender](https://i.loli.net/2020/04/01/itcNXSEYTCfVehd.png)



## 解决方案



一种方法，进入 **Windows 安全中心**，看到一个盾牌，点进去，进入**管理设置**



![fuck-wondows.jpg](https://i.loli.net/2020/04/01/Sl4auje8wnFiLGC.jpg)



看到**实时保护**，将它关闭，这样就可以在一段时间内免遭 WD 乱删文件，不过这狗比玩意好像是下次开机会自己启动的，所以好像还治不了根，目前好像只有这种方法了，再乱删的时候就再关闭就好了，这点 Windows 真的好狗



![close-prot.jpg](https://i.loli.net/2020/04/01/RauzUSW8If4OZLC.jpg)



如果觉得这玩意一段时间后还是重启乱删东西很不爽的话，接着看下去：同是在**管理设置**里面，往下翻，有一个**排除项** ，点开来



![exclude](https://i.loli.net/2020/04/04/s6aFx9oBnEDGj2P.png)



将可能会被误删的东西放到一个文件夹里边儿，然后将这个文件夹添加进**排除项**，以后它应该就不会删除这里面的东西了（但愿）



![exclude-folder](https://i.loli.net/2020/04/04/X1JjyuTHpLtCW7S.png)