---
layout:     post
title:      BUUCTF Writeup  Misc
subtitle:   Continuous updating
date:       2019-11-21
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - writeup
    - security
---



## preface



本文是 BUUCTF 杂项题的 Writeup，持续更新中



## 金三胖



下载后是一张 gif 图，隐隐约约看到了 flag ，直接用 StegSolver 的 Analyse-> Frame Browser对 gif 进行逐帧观察即可获得 flag



![jinsanpang.jpg](https://i.loli.net/2019/11/25/xXb8hnfeGps9tuZ.jpg)



> flag: flag{he11ohongke}



## 二维码



只有一个二维码，扫描之后也没啥信息，显示 `secret is here` ，拖到 binwalk 里面发现除了图片外还有个压缩包在里面



![erweima.png](https://i.loli.net/2019/11/25/cxjaGPDVsACUFqZ.png)



用 foremost 解压缩，发现压缩包被加密过，里面有个 txt 文件名叫 4number ，猜想用弱口令爆破密码



![4number.jpg](https://i.loli.net/2019/11/25/M5Ocw19BQEifn7p.jpg)



爆破成功，拿到密码，解压缩包



![erweima-baopo.jpg](https://i.loli.net/2019/11/25/D7WiH8f9RdLXU1B.jpg)



点进解压出来的 txt 文件即可拿到 flag



> flag: flag{vjpw_wnoei}



## N 种方法解决



得到一个 windows .exe 后缀的可执行文件，拖到 WSL 里面用 file 分析一下，本质上是 ASCII 文本，被换了格式，cat 输出文本，这里面是一张图片的 Base64 编码过后的内容，直接复制到在线 Base64 转图片平台还原



![Nzhong.jpg](https://i.loli.net/2019/11/25/x2OYqDNG8m1Vlj6.jpg)



得到一张二维码，手机扫描即可拿到 flag



![Nzhong2.jpg](https://i.loli.net/2019/11/25/nSOkYWeqjT37Dav.jpg)



> flag: flag{dca57f966e4e4e31fd5b15417da63269}



## 基础破解



zip 包，提示加密的密码为 4 个数字，直接爆破，得到密码



![baopo.jpg](https://i.loli.net/2019/11/25/pfAd31iKXTlIJPc.jpg)



打开压缩包后是一个 txt 文件，里面是一段 Base64 加密过的文本，直接解码得到 flag



> flag: flag{70354300a5100ba78068805661b93a5c}



## 大白



下载到一张大白的图片，根据题目意思应该是被截断过的，打开 winhex 查看是 png 格式的图片，查资料，IHDR 块后面的八个字节分别代表了图片的宽和高



![ihdr.jpg](https://i.loli.net/2019/11/25/a9nyZAGWEgwK7UV.jpg)



编辑图片的高度，使 flag 出现



![dabai.jpg](https://i.loli.net/2019/11/25/FCuilp3W5Qfv6Z1.jpg)

>  flag: flag{He1l0_d4_ba1}



## 你竟然赶我走



直接将图片用 winhex 打开拉到最底下得到 flag 



![ganwozou.jpg](https://i.loli.net/2019/11/25/gD5d6zPrNcYesjv.jpg)



> flag: flag{stego_is_s0_bor1ing}



## rar



题目说了，加密的压缩包的密码是四位数字，直接先爆破拿到密码



![rar.jpg](https://i.loli.net/2019/11/25/opBEHhw6MqOuKgY.jpg)



解压缩包，得到 txt 文件里面就是 flag



> flag: flag{1773c5da790bd3caff38e3decd180eb7}



## qr



签到题，直接扫二维码就出 flag



> flag: flag{878865ce73370a4ce607d21ca01b5e59}



## 文件中的秘密



直接把图片拖到 winhex 下得到 flag



![file-secret.jpg](https://i.loli.net/2019/11/25/zIhdue6lTBK1YmD.jpg)



> flag: flag{870c5a72806115cb5439345d8b014396}



## wireshark



题目提示说是管理员登录网站的流量包，那判断应该是 http 协议，一般登录提交表单用的是 POST 方法，所以过滤一下就得到了 flag



![wireshark.jpg](https://i.loli.net/2019/11/25/YUCw6qkzlpvmBEa.jpg)



> flag: flag{ffb7567a1d4f4abdffdb54e022f8facd}



## 另外一个世界



将图片拖到 winhex 里，翻到最下面是一串二进制数



![another-world.jpg](https://i.loli.net/2019/11/25/2G6opItb9YjOFzX.jpg)



将二进制数转化成字符串，得到 flag



> flag: flag{koekj3s}



## 





## 被嗅探的流量



过滤 http POST 流量，有两条记录，点进其中一条可以看到上传了 flag.jpg ，追踪 http 流，然后搜索 flag ，在最下面可以找到



![流量.jpg](https://i.loli.net/2019/11/25/aRikH5egsVFL9hx.jpg)



> flag: flag{da73d88936010da1eeeb36e945ec4b97}



## 来首歌吧



这题下载下来是个音频文件，打开听没什么不正常的，猜想是音频隐写题，在 windows 下我们一般用 Audacity 打开音频文件，因为里面可能隐藏着些东西，果真，打开来就可以看到摩斯密码，直接解密就拿到了 flag（耗时比较久，解密太麻烦了）



![mors.jpg](https://i.loli.net/2019/11/25/qCN3E4HUSJMn9YQ.jpg)



>  flag{5BC925649CB0188F52E617D70929191C}



## easycap



打开流量包，选择握手包，追踪 TCP 流，拿到 flag



![easycap.jpg](https://i.loli.net/2019/11/25/kZrxgjva89Atuhe.jpg)



> flag: flag{385b87afc8671dee07550290d16a8071}