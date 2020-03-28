---
layout: post
title: GitHub page接入Google广告
subtitle: 
date: 2020-03-27
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - blog
---





## preface



今天突然上 Google-Analytics 看了看，发现我的博客突然多了很多流量，我擦，写的这篇路由器翻墙的文竟然还有这么多人看，还是挺欣慰的，这个小破站建立已经一年多了，不比在大平台写博客，GitHub pages 搭的个人博客基本没有什么推广的，能看到都是别人在浏览器里面搜到的，顶多通过友链进来，而且百度还不收录 GitHub page 搭的网站，因此流量更是少得可怜，稍微有人看一下都很高兴了。



![Google-analysis](https://i.loli.net/2020/03/28/TJ5DlApf36HXnLd.jpg)



Google 搜了一下，我这篇文章竟然排在第一位，惊到我了，好吧，以后多多输出优质内容，虽然写博客初衷是作为自己的笔记本的，但是写的内容还是得要对得起大家才行，为了促使自己多写点博客，我想在网站中加入一些广告，通过金钱的诱惑使自己生产力提高（才不是）



![openwrt](https://i.loli.net/2020/03/28/XpndHVQzBAPI2UL.jpg)



投放广告还是选择 Google 家的广告了，大牌，放心，而且在很多网站都可以看到 Google 的广告，比如最近我迷上的 **莫烦python** 的[网站](https://morvanzhou.github.io)，在网上搜到一个程序媛姐姐写的[教程](http://codewithzhangyi.com/2019/11/06/google-adsense/)，挺不错的，参考一下，我也为我自己用 GitHub page 搭建的网站加入 “谷歌联盟” Google AdSense，下面是具体做法：



## 注册 Google AdSense



首先我们要去 Google AdSense 注册一个账号，网站链接[点这里](https://www.google.com/adsense)，然后填入自己网站的域名和相关信息，邮箱就填自己的 Gmail 吧，后续也方便用自己的 Google 账号登陆，完事之后就会给我们一串 js 代码，复制到网站里面，我将它插入到了 `head.html` 的 `<head>` 标签中（放到 body 中也可以）



```html
<script data-ad-client="ca-pub-8559254398024xxx" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
```



然后在 Google AdSense 中点击 **我已插入代码**，它就会去我们的网站上查找这段代码，找到之后就成功了第一步，接下去就是审核网站，可能时间会比较久，耐心等吧，审核结束也会发邮件给我们



![verify](https://i.loli.net/2020/03/28/loJW8C1IpvAqRTX.jpg)



## 审核完再更新

