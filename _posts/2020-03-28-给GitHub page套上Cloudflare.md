---
layout: post
title: 给GitHub page套上Cloudflare
subtitle: 基于万网域名
date: 2020-03-28
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - blog
    - computer network
---



## preface



大家都知道 GitHub page 上的博客是基于 GitHub 服务器搭建的，虽然 GitHub 非常慷慨，给了我们很大的容量和流量，但是毕竟服务器在美国，所以国内的访问速度还是比较慢的，其实挺想把博客移植到我的阿里云学生机上，以后再说吧，目前最方便的方式就是给博客套一层 Cloudflare 来加快访问速度



## Cloudflare



> **Cloudflare** 是一间总部位于旧金山的美国跨国 IT 企业，以向客户提供基于[反向代理](https://zh.wikipedia.org/wiki/反向代理)的[内容分发网络](https://zh.wikipedia.org/wiki/內容傳遞網路)（Content Delivery Network, CDN）及分布式域名解析服务（Distributed Domain Name Server）为主要业务。利用Cloudflare全球Anycast网络，Cloudflare可以帮助受保护站点抵御 DDOS 等网络攻击，确保该网站长期在线，同时提升网站的性能、加载速度以改善访客体验。
>
> 截至2020年1月，Cloudflare拥有200多个位于全球各地的数据中心



总的来说。就是一家给用户提供全球 CDN 节点以及 DNS 服务的公司，他好在哪里呢，前面说了，Cloudflare 是基于反向代理的 CDN，也就是说用户在访问套有 Cloudflare 的网站时访问的并不是网站真实所在的服务器，而是一台反向代理服务器，代理服务器将请求转发给真正的服务器，然后将相应返回给用户，这样就可以达到隐藏服务器真实 ip 地址的目的，更重要的是可以防止来自网络上的攻击，这也正是 Cloudflare 这家公司的目的。并且使用 Cloudflare 的 CDN 是免费的，那还等什么，肯定要冲啊



![reverse-proxy](https://i.loli.net/2020/03/28/zAabZoI1XhsitlK.png)



## 内容分发网络 CDN





> 内容分发网络或内容分发网络(CDN)是由代理服务器及其数据中心组成的地理分布式网络。目标是通过相对于最终用户在空间上分布服务来提供高可用性和性能。CDNs出现于20世纪90年代末，是缓解互联网性能瓶颈的一种手段。



简单来说，CDN 就是部署在世界各地的缓存服务器，它们会提前缓存网站上的资源，然后当用户想要访问相关资源时，直接从 CDN 服务器上取就可以了。这样不仅可以增加访问速度减少访问延迟，还可以减缓网站服务器上的压力。根据 CDN 的机制，我们在访问服务器时会访问到一个离我们最近的 CDN 服务器，上面已经缓存了服务器的内容，所以这样可以大大提高网页的访问速度。



值得一提的是，自己购买 CDN 服务器的话非常昂贵，国内有很多公司也提供了 CDN 服务，价格也不菲，Cloudflare 在中国也有 CDN 服务器，并且还可以免费使用，对于个人博客是够用了（下图为使用单个服务器与使用 CDN 的区别）



![img](https://upload.wikimedia.org/wikipedia/commons/f/f9/NCDN_-_CDN.png)





>  注意: 本篇文章基于 GitHub page ，如果是国内的网站的话，最好不要套 Cloudflare，可能会使得访问速度变得更慢



## 注册 Cloudflare



想要给 GitHub page 套上 Cloudflare 我们需要两样东西，一样是个人域名，一样是 Cloudflare 账号。我之前已经买了一个域名并且解析到博客地址，所以这一步就省了，没有域名的自己去买一个吧，便宜的很。所以我们接下来就搞搞 Cloudflare



点击注册填完信息之后，网站会问我们有什么计划，直接选择免费的就行了，适用于个人博客

![plan](https://i.loli.net/2020/03/28/63dPw5FTzYL9glp.jpg)

接着填入自己网站的域名（个人域名，不要填 xx.github.io），网站会自动扫描当前域名的解析记录

![scan-dns](https://i.loli.net/2020/03/28/Vf2yYliu3KQcthU.jpg)

确认没什么问题的话就可以点击 `continue` 继续下一步

![records](https://i.loli.net/2020/03/28/6WlUdT5rSwRXgoV.jpg)



## 修改 DNS



然后官方就会要求我们修改域名的 DNS 服务器，下面给出了我这个域名目前的 DNS 服务器以及官方要求我们改的 DNS 服务器，按要求更改就行了



![change-dns](https://i.loli.net/2020/03/28/8fme2pxCQdbVaRG.jpg)



我的域名是在阿里云的万网上面买的，更改 DNS 服务器的路径为 **域名服务-域名列表-管理**



![config-domain](https://i.loli.net/2020/03/28/LZ3uXMEIqvWN6Qf.jpg)

​	

然后就可以看到当前域名的具体信息，在 DNS 服务器哪里点击修改 DNS ，将现在的 DNS 服务器换成 Cloudflare 的 DNS 服务器保存即可



![aliyun-dns](https://i.loli.net/2020/03/28/ZDCv3OSHQtLIEqc.jpg)



改完了要过一会儿等 DNS 服务器生效后，官方就会发送信息到邮箱里说域名解析成功，见到以下界面就说明我们的博客已经在 Cloudflare 的保护之下了



![success](https://i.loli.net/2020/03/28/4DS8vdsiNwnOG3V.jpg)

最后不要忘了将 **Always Use HTTPS** 勾选，这让我们的网站强制使用了 HTTPS 通信，减少了数据被嗅探的可能性。而且 Cloudflare 默认是双端加密的，从浏览器到 Cloudflare 以及从 Cloudflare 到网站服务器都是经过了 HTTPS 加密通信



![force-https](https://i.loli.net/2020/03/28/l8bQXaHZSTmjo6C.jpg)



我们来看看是否生效，查询一下我们的服务器域名解析的 ip 和 xx.github.io 解析的 ip

```bash
$ nslookup szukevin.site

Name:      szukevin.site
Address 1: 104.28.31.244
Address 2: 104.28.30.244
Address 3: 2606:4700:3037::681c:1ef4
Address 4: 2606:4700:3036::681c:1ff4
```



```bash
$ nslookup yarkable.github.io

Name:      yarkable.github.io
Address 1: 185.199.109.153
Address 2: 185.199.108.153
Address 3: 185.199.111.153
Address 4: 185.199.110.153
```



xx.github.io 解析到的 ip 都是后面四个，这是 GitHub page 的服务器，而上面我们用了 Cloudflare 服务的域名解析出来的 ip 已经变了，这就是反向代理的功劳，隐藏了服务器真实 ip 地址。到这里我们的博客就已经套上 Cloudflare 服务了，不挂代理的访问速度确实快了些（应该不是心理作用）



## attention



完事之后关于域名要注意一些事项，由于我们已经将 DNS 服务器改成 Cloudflare 的 DNS，所以之前解析过的域名都无效了，需要在 Cloudflare 里面重新进行域名解析，不能在之前的域名提供商那里进行解析，要在 Cloudflare 的 DNS 选项卡里面进行解析。



这里给出我自己的解析表，顺便说说里面的 **Proxy status** 代表的含义。可以看到我一级域名是直接解析到 GitHub page 的，所以状态为 **Proxied** ，表示经过 Cloudflare 代理访问以达到 CDN 加速和伪装 ip 的功能。有几个二级域名解析到了公网，是我用来做实验的，并没有网站架设在服务器上，所以我不需要 Cloudflare 代理，只要它提供 DNS 解析的功能就行了，所以状态是 **DNS only** 。还有几个二级域名解析到了内网，同样是我拿来做实验的，我只需要 DNS 服务器解析域名就行了，但是由于 ip 不是公网，无法直接访问，涉及到端口转发等概念，因此状态为 **DNS only - reserved IP** （reserved IP 就是局域网 ip 的意思）



![status](https://i.loli.net/2020/03/28/kJibEgpjeYxNlsK.jpg)



---



假设我们将 aliyun 这个二级域名的状态变成 **Proxied** ，会发生什么呢，名场面，相信很多人都见过，一看图就明白了，Cloudflare 代理了浏览器对 aliyun.szukevin.site 解析到的公网地址的访问，但是由于我这台公网服务器的 80 端口没有假设网站，所以 Cloudflare 到服务器这条路不同，而浏览器客户端到 Cloudflare 是连通的



![proxy-error](https://i.loli.net/2020/03/28/35PavW2merDzxVO.jpg)



如果只是 **DNS only** 的话就会显示网站访问失败，因为这个服务器上确实没有网站，这就没有经过 Cloudflare 的代理直接到达网站真实 ip 地址。然后还有个点，由于我们强制用了 HTTPS ，上面 **Proxied** 的状态时因为用了 Cloudflare 的服务，所以访问域名时即使出错了也是用 HTTPS 加密的，而 **DNS only** 就只有不安全的 HTTP 通信了



![non-proxy](https://i.loli.net/2020/03/28/puwdPzexSD896vi.jpg)



我还发现用 Cloudflare 解析记录的话挺慢的吧，有时候可能过个几分钟才解析生效，看是否生效就在命令行 ping 一下域名吧，一个很神奇的事，WSL 在 ping 的时候经常会报错找不到域名，而用 windows 自带的 powershell 去 ping 就能够很快知道 ip 地址（更新：后来发现是我的 WSL 坏了…）



![ping-domain](https://i.loli.net/2020/03/28/4ZlUSh2BbVa6o5t.jpg)

