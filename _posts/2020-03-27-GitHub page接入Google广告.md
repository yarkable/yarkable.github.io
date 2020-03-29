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



Okay，经过了漫长的几个小时后，它审核完了，我现在可以去设置广告了！



![serve-ads](https://i.loli.net/2020/03/28/JViyN8kFTrG3eDA.jpg)



然而，他又说为了不影响我的收入(不存在的)，让我下载一个 `ads.txt` 文件，其实里面的内容就是广告商的名字和我的广告 ID 号等等，直接在网站根目录新建一个 `ads.txt` ，把内容复制进去就好了



![ads-txt](https://i.loli.net/2020/03/28/VgC4KRm1nfDTBpv.jpg)



然后访问网站根目录 `/ads.txt` 如果出现了一串神秘的文本就说明这一步已经 OK，虽然这时候网站可能还是会提醒说我们没有 `ads.txt` ，不用着急，可能要几天才能生效

```txt
google.com, pub-85592543xxx4365, DIRECT, f08c47fec094xxx0
```



## 选择广告



接着我们就可以生成广告了，这里的广告有两种，一种是自动广告，Google 会自动在网站合适的位置添加广告，我试了一下，这广告太多了，而且还很大，算了，自己都看不下去。还有一种就是按广告单元分的广告，这又分为三种：**展示广告**，**信息流广告**，**文章内嵌广告**。emm，看了一下，好像没什么差别？？



![adsssss](https://i.loli.net/2020/03/28/oWxzlADqvnidbRt.jpg)



最终选择了**文章内嵌广告**，可以自己设置广告样式和尺寸，不过，不想花精力去搞这种东西哈



![third-ads](https://i.loli.net/2020/03/28/wbXmqgpRsfCJ4zQ.jpg)



保存之后就生成了一段 js 代码，复制到网站的  `_post.html`  (每种博客模板的文件名可能不一样)，然后下次别人来访问网站就能够看到展示的广告了。

```html
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<ins class="adsbygoogle"
     style="display:block; text-align:center;"
     data-ad-layout="in-article"
     data-ad-format="fluid"
     data-ad-client="ca-pub-8xx925xxx4365"
     data-ad-slot="5744645800"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
```



## 后记



然后我就把广告贴在了最底下，不影响大家看文章，不过有时候广告并不会出现，出现的是一大片的空白区域，而且不挂代理的时候广告会损坏，毕竟中国把 Google 的应用给墙了……

![my-workout](https://i.loli.net/2020/03/28/uyTXRBPWk7Jc6oN.jpg)



早知道不搞这玩意了，瞎几把折腾，又没多少钱，还碍眼，罢了罢了，等下次换 Hexo 主题的时候就把这广告给弄走，本来这主题还挺清秀的，加了个广告变得太杂了，想给博客弄广告的朋友们，给个忠告，还是算了吧，别搞了，真的没几把用

