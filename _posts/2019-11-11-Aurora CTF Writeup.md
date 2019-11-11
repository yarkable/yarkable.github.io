---
layout:     post
title:      Aurora CTF Writeup
subtitle:   I am so vegetable :D
date:       2019-11-11
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - writeup
    - security
---



## preface



这是深大 Aurora 战队招新题的 writeup ，本菜鸡饱着学习的态度参加比赛，记录下自己的解题思路



## web



### PHP is very good



打开题目，就是一张你的名字 == ，查看网页源码，说源码在 /code 里面但是 flag 就在这个页面中



![](C:\Users\kevin\Desktop\blog\aurora_php_is_very_googd_1.jpg)

那就进入 /code 康康，是一段 php 代码，分析一下，是要用 `GET` 方法向 /code 中传入一个 `code` 字段，code 要先经过一段正则过滤，然后再经过一段正则匹配，最终让 `eval()` 函数把 code 当成 php 代码来执行，因此，应该是考察 eval() 函数的漏洞



![](C:\Users\kevin\Desktop\blog\aurora_php_is_very_googd_2.jpg)

先分析一下第一个正则表达式，之前一直没见过 `(?R)` 这种表达，上网搜才知道这是 php 的递归正则

```php
/[a-z]+\((?R)?\)/
```

因此要想通过第一个正则，我们应该构造出一个若干小写字母后面加上括号(括号中又可以重复这些内容)的字段，并且除此之外要有一个分号，以下字段都是可以通过第一个 if 判断的：

* demo();
* demo(demo());
* demo(demo(demo()));



发现第二个正则前面没有加上 `/` ，本来应该是要和后面的 `/` 配套的，就不管他，想着要构造什么才能让 eval() 输出我们想要的东西，只想到 `phpinfo()` ，输进去确实可以出来结果，但是找了半天也没发现什么是和 flag 有关的，也没查到资料，就这样吧，做了好久这题，看了 wp 再来更新



![](C:\Users\kevin\Desktop\blog\aurora_php_is_very_googd_3.jpg)



### checkin



查看源码，发现注释处有段奇怪的串串，看后面有两个 == 应该是 base64 加密过的 flag，拖进 burp解密，得到 flag



![](C:\Users\kevin\Desktop\blog\aurora_checkin.jpg)



### Welcome to Aurora



![](C:\Users\kevin\Desktop\blog\aurora_welcome_to_aurora.jpg)



这题特效可以玩一年哈哈哈，在 bugku 上看过这个模板出的题，这题介绍上写着加入战队获取 Aurora 浏览器，查看一下 cookie ，上面有个 `member=false` ，我以为就是很 ez 地改一下 cookie=true 就能拿到 flag 了，结果啥也没有，试了各种方法，还以为 cookie 修改错误了，最终放弃 cookie



![](C:\Users\kevin\Desktop\blog\aurora_welcome_to_aurora2.jpg)



转眼一想，使用 Aurora 浏览器访问，应该是要修改 `User-Agent`，上网搜索 Aurora 浏览器地 User-Agent ，还真有 Aurora 浏览器，但感觉应该没有这么复杂，于是就简单的修改了一下 User-Agent ，就出 flag 了，所以这题要改的地方有两个，一个 User-Agent，一个 cookie



![](C:\Users\kevin\Desktop\blog\aurora_welcome_to_aurora3.jpg)



### ez LFI



一道简单的 php 本地文件包含题，上网搜索一下基本原理后就可以直接用 php 伪协议读取源码了

```cpp
http://aurora.52szu.tech:5006/index.php?file=php://filter/read=convert.base64-encode/resource=flag.php
```



读到的是 base64 加密后的源码，不加密的话会被当成 php 代码执行，因此只需要解密就行了



![](C:\Users\kevin\Desktop\blog\aurora_ez_lfi.jpg)



### ssti



开始看这题没啥思路，好像不会对输入进行检查，随便输入什么都行， F12 也没啥有用的信息，再看下题目，Google 一下 ssti ，应该是道模板注入题，之前做过一题 flask 的模板注入，应该是差不多的道理：输入模板，他会把模板中的内容当成 python 代码运行，输入 `{{ 1 + 1 }}` ，然后验证了猜想



![](C:\Users\kevin\Desktop\blog\aurora_ssti.jpg)

![](C:\Users\kevin\Desktop\blog\aurora_ssti2.jpg)



## RE



### re_signup



把培训时的 PPT 翻出来看了一下才去做题的，这题直接拖进 IDA 里面就可以找到答案了，虽然的确有点麻烦，真的是一个一个找的，期间才学会 IDA 的







## MISC



### vim2048



不多说，也就玩了几个小时吧，然后 vim 更加熟练了，嗯（找不到 flag 图了，拿一张失败的凑数）



![](C:\Users\kevin\Desktop\blog\vim-2048.png)



### base64_stego



这题下载下来是一串东西，从最后的两个 == 可以看出来就是 base64 加密过的



![](C:\Users\kevin\Desktop\blog\aurora_base64.jpg)



拖去解密，大意就是说隐写术巴拉巴拉的，这里注意，有些在线的 base64 解码只能解一句话，后面的话不会显示，有点坑，我这是在 burp 里面解码之后的输出



![](C:\Users\kevin\Desktop\blog\aurora_base64_2.jpg)



然后就上网搜索 base64 隐写，有很多相关描述，具体的原理等以后专门写一篇来讲一下吧，这里直接贴上脚本







### san check



真 签到题，去首页复制 flag 就行了



### 网线鲨鱼



做这题之前还特意看了一下 wireshark 教程，发现这玩意真有意思，书上那么枯燥的概念在 wireshark 里面一目了然，回归正题，直接在网络包里过滤 HTTP 请求，然后找到一条对 OJ 的 POST 请求，猜想应该就能拿到密码了，果然如此



![](C:\Users\kevin\Desktop\blog\aurora_wlan.jpg)



### 网线鲨鱼 EVO Plus



这题猜想应该是让我们破解 WIFI 密码，整个流量包都是 802.11 协议，出现了几个 EAPOL 协议的包，上网搜，说这个是握手包，那就过滤出来，一共是 4 个握手包



![](C:\Users\kevin\Desktop\blog\aurora_wlan_plus.jpg)



既然握手包都有了，那么接下去完全就可以破解密码了，开启 `aircrack-ng` 冲，然而用了七个字典都没有破解出密码，可能这个密码有点复杂或者是这些字典还不够牛逼



![](C:\Users\kevin\Desktop\blog\aurora_wlan_plus2.jpg)



然后想起来，之前的题目里好像有出现过密码，说不定这里可以直接用，找到之前的包，过滤一下找到密码



![](C:\Users\kevin\Desktop\blog\aurora_wlan_plus3.jpg)



去 wireshark 里面设置一下，如果知道 SSID 和 密码就能解密了，然而并没有出现新的流量，所以应该密码错了，行 8 ，这题就先这样吧，等看了 wp 再更新 



![](C:\Users\kevin\Desktop\blog\aurora_wlan_plus4.jpg)