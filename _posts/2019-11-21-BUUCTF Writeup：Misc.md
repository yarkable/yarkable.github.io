---
layout:     post
title:      BUUCTF Writeup：Misc
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



## 乌镇峰会种图



直接 winhex 拖到最后拿到 flag



![wz.jpg](https://i.loli.net/2019/11/25/kDKPZ8wcTqeRax4.jpg)



>  flag: flag{97314e7864a8f62627b26f3f998c37f1}



## LSB



拖到 stegsolver 里面用 `Data Extract` 来提取最低位，发现藏着一张 png 图片



![lsb.jpg](https://i.loli.net/2019/11/27/AmS4aW3uIdh1gOY.jpg)



点击 `save bin` 保存为 png 格式，得到一张二维码，扫描得到 flag



![image.png](https://i.loli.net/2019/11/27/vEG1uZcPF6SLyke.png)



> flag: flag{1sb_i4_s0_Ea4y}



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



## ningen



常见的图片隐写题，在 winhex 中发现图片里面有 ningen.txt 字样



猜测应该是隐藏了东西，将图片拖到 binwalk ，foremost 分离出一个加密的 zip 包，根据提示 zip 包的密码是 4 位数字，直接爆破得到 flag



> flag: flag{b025fc9ca797a67d2103bfbc407a6d5f}



## 你竟然赶我走



直接将图片用 winhex 打开拉到最底下得到 flag 



![ganwozou.jpg](https://i.loli.net/2019/11/25/gD5d6zPrNcYesjv.jpg)



> flag: flag{stego_is_s0_bor1ing}



## rar



题目说了，加密的压缩包的密码是四位数字，直接先爆破拿到密码



![rar.jpg](https://i.loli.net/2019/11/25/opBEHhw6MqOuKgY.jpg)



解压缩包，得到 txt 文件里面就是 flag



> flag: flag{1773c5da790bd3caff38e3decd180eb7}



## 假如给我三天光明



得到一张图和一个加密过的压缩包，图片里面可以看到八个盲文



![helen.jpg](https://i.loli.net/2019/11/25/QFvDJP43s6198bx.jpg)



通过对应关系找到密码为 `kmdonowg` ，破解压缩包拿到一段音频，用 Audacity 打开，又是摩斯密码。。



![monalisa.jpg](https://i.loli.net/2019/11/25/dZrW1uAkfeUzMvn.jpg)



网上搜到一个[很牛逼的东西](https://morsecode.scphillips.com/labs/audio-decoder-adaptive/?tdsourcetag=s_pcqq_aiomsg)，将带有摩斯密码的音频直接拖进去就可以得到密码(题目好坑，最后的 flag 是小写的)



> flag: flag{wpei08732?23dz}



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



## snake TODO



得到一张蛇的图片，拖进 binwalk 发现有个压缩包，用 foremost 分离出来







提取压缩包中的文件，是一段 Base64 加密过的文本和一个二进制文件，解密 Base64 之后得到一段话



``` 
What is Nicki Minaj's favorite song that refers to snakes?
```



搜索这个人，然后就找到了她唱过 anaconda ，flag 应该就是这个，干，错了。然后看看文件夹中另外一个 cipher 文件，又不是二进制文件，用 file 命令看看是个数据文件



![cipher.jpg](https://i.loli.net/2019/11/25/Jgwm54zoWpOKsUd.jpg)



真的想不到能对它做什么，就去看了 wp ，擦，我傻了，思维定势，原来这里也牵扯到密码学的内容了，我们刚刚得到的 anaconda 就是 KEY ，是个公钥，cipher 是密文，因此应该要用某种解密方法去得到明文，



![decry.jpg](https://i.loli.net/2019/11/25/ZliOfQE5ATLPnsz.jpg)



## 小明的保险箱



binwalk 分析，图片藏着一个压缩包，foremost 分离，压缩包加密，题目说了密码是 4 位数字，直接爆破



![xm.jpg](https://i.loli.net/2019/11/25/6EIvm4zco3qpeVJ.jpg)



> flag: flag{75a3d68bf071ee188c418ea6cf0bb043}



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



## 爱因斯坦



这张图拖到 binwalk 里面分析，藏着一个压缩包，foremost 分离，想解压，发现要密码，然后在图片的详情中找到了一个备注，开始我以为这真的不是密码，放进去试了一下，真就解开压缩包密码了。。直接拿到 flag



![Einstein.jpg](https://i.loli.net/2019/11/25/ZXmKF8bGjDWcB7U.jpg)



> flag: flag{dd22a92bf2cceb6c0cd0d6b83ff51606}



## 隐藏的钥匙



这题有点东西啊，开始把放进 winhex 没啥情况，再将图片拖进 binwalk，发现里面还藏着一张图片，用 foremost 分离，以为 flag 在第二张图片上面



![key.jpg](https://i.loli.net/2019/11/25/oCkIwlt1N59jgRK.jpg)

找了半天找不到 flag ，无奈看 wp ，淦，原来在第一张图片的 hex 部分直接可以找到 Base64 加密过的 flag ，怪我没仔细看，不过这也太坑了吧，那干嘛要搞两张图片出来啊！



![base64](https://i.loli.net/2019/11/25/WN2VmAU7CMSsh1Y.png)



> flag: flag{377cbadda1eca2f2f73d36277781f00a}



## 弱口令 TODO



题目提示是弱口令，然后破解了好久没打开压缩包，看了网上的 wp ，擦，直接打开压缩包可以看到注释



![ruokouling.jpg](https://i.loli.net/2019/11/27/nOIoZE5fW82vGgi.jpg)



虽然啥都没有，但是复制到 sublime 里面就看到了点东西，摩斯密码在线解密得到密码为 `HELL0FORUM` 



![ruokouling2.jpg](https://i.loli.net/2019/11/27/sUF7lr4KbjXgfJY.jpg)







## 荷兰宽带数据泄露



这题直接给了个 `conf.bin` 文件，里面是二进制内容，打不开，又没有什么提示，宽带数据是啥，网上看 wp 才知道这是个路由器的配置文件，WTF ？？ 这是怎么知道的，好吧，做多了应该就知道了，用一款叫做 [RouterPassView](https://www.nirsoft.net/utils/router_password_recovery.html#DownloadLinks) 的软件可以打开这个 bin 文件，找到用户名和密码，这题一点 hint 也没有，都不知道提交的是啥，其实原题是有说提交的 flag 是配置中的 username 



![helan.jpg](https://i.loli.net/2019/11/25/nJf7UVPG6p2kvwj.jpg)



> flag: flag{053700357621}



## 神秘龙卷风



弱口令暴力破解压缩包，解压缩文件里面得到这样一段东西

















