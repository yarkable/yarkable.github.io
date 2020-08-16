---
layout: post
title: BurpSuite抓IOS设备HTTPS流量
subtitle: 
date: 2020-08-16
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - web
    - security
    - 
---



## preface



Burp 这个工具做过 web 安全的人都应该用过，是个非常强大的抓包工具。在 PC 的浏览器上直接配置代理就行了，本篇文章就来介绍一下如何用 Burp 抓 IOS 设备上的流量，很多文章都介绍过怎么抓包，但是很多坑都没有说到，这里一些要避免的坑我都记录了下来。



## 常规步骤



第一步，首先要在电脑上开启 Burp，并且将监听的地址选择成 `LocalIP`（也就是 ipconfig 出来的那个 ip，并不是本地环回），在 Burp 上下拉就可以知道当前 IP 是多少了 ，端口自己随意选择



![burp](https://i.loli.net/2020/08/16/Buny3tzmc2ZDxWv.png)



之后在 IOS 端（我是 iPhone）连上和 PC 同一个网段的 WIFI ，再手动配置一个代理，服务器就填 PC 的 IP，端口也和 PC 端 Burp 监听的端口一样，然后确定即可。



![WiFi](https://i.loli.net/2020/08/16/xRBe4WFzIHYpvKs.png)



之后在 safari 中访问 `192.168.1.200:8081` （每个人配置不一样），右上角就会有一个按钮，点击一下就会让我们下载 Burp 的证书，这是为了抓 HTTPS 流量用来验证的证书。然后我们要去 IOS 的设置处去信任该证书。



![cert](https://i.loli.net/2020/08/16/Cc8uJ2Eb4XZswrg.png)



如果一切正常的话，只要信任了该证书，那么我们在手机上访问网络的话，在 PC 端的 Burp 就可以看到相应的请求被拦截下来，这里是我拦截的一个微信公众号的表单，可以看到，POST 请求的参数以及 cookie 等都可以被抓到，跟在浏览器上抓包是一样的。



![crawl](https://i.loli.net/2020/08/16/myx5poXKDAGhPjO.png)



## Troubleshoot



### 下载不了证书



上面的内容网上的很多文章都可以找到，但是他们没有写发生问题该怎么做。开始时我已经按照上面的步骤走完了，惊奇的是，我连证书都下载不下来，更加别说抓包了，就连 HTTP 请求都不能被 Burp 抓到。然后就一直在那里找问题所在，试了好久，最终终于知道了……



第一个可能的因素就是**手机开了代理**，不过之后我在成功之后再将代理打开也可以抓到包，所以应该不是这个原因，总之，如果不行的话，就尝试将代理关闭，PC 端的代理也关闭试试。



第二个因素，就是这个让我连不上 Burp，那就是 **Windows 防火墙**，直接将防火墙全部关闭就可以抓到包了，被这坑死……

```txt
控制面板 -> Windows defender 防火墙 -> 启用或关闭 Windows defender 防火墙 -> 专用和公用网络全都关闭
```



### 抓不了 HTTPS 请求



解决了上面的问题之后，我尝试抓微信的包，还是不行，拦截不了，然后在 Burp 的 Dashboard 看到了这个提示



![log](https://i.loli.net/2020/08/16/uFZH1b9B3ga6MTR.png)



上网搜索之后找到了答案，原来 IOS10 之后仅仅信任证书还是不够的，要在 `设置-通用-关于本机-证书信任设置` 里面对根证书进行完全信任才行。然后就可以抓到 IOS 设备上的 HTTPS 流量了，害，应该没有什么别的坑了。



![setting](https://i.loli.net/2020/08/16/Rd7mS4n95fh3wN2.png)



## reference



https://portswigger.net/support/configuring-an-ios-device-to-work-with-burp

https://forum.portswigger.net/thread/ios-8-the-client-failed-to-negotiate-an-ssl-connection-to-47727c92

