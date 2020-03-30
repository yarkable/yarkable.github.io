---
layout: post
title: Typora自动上传图片至smms图床
subtitle: 以及一系列填坑
date: 2020-03-24
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - markdown
    - troubleshoot
---



## preface



前段时间听说 typora 新版本内置了 PicGo 工具，可以直接上传图片到图床了，这可真是太方便了，之前我写博客就用的是 PicGo 工具，也挺方便的，粘贴图片路径就直接返回 markdown 链接，不过感觉速度稍慢，而且还是要自己将链接复制到 typora 中，有了新版本的 typora 的这个功能，我们直接将图片贴到 typora 中就不用管了，后台自动上传。



## 配置 typora



首先得将 typora 更新到最新版本，其次 PicGo 也得更新到最新版，我之前用的是 2.1.2 版本，最新的是 2.2.2 版本，不更新的话是用不了这个功能的，PicGo 的下载链接[在这里](https://github.com/Molunerfinn/PicGo/releases/tag/v2.2.2)，不翻墙的话速度很慢



然后打开 typora 的设置，找到偏好中的图像一栏，像下面这样改，当插入图片的时候就直接上传图片，上传服务选择 PicGo，然后将路径添加上去，最后点击一下验证图片上传选项

![setting](https://i.loli.net/2020/03/27/VLyjlMEuog7zqZr.png)



如果和我一样的话，就说明已经配置成功了，接下去就可以愉快的玩耍了

![](https://i.loli.net/2020/03/27/LAPIig9Em4eNHsx.png)

但是绝大多数人都还不能够，因为没有配置 PicGo，简单来说 typora 这个功能的原理就是启动 PicGo 客户端并且通过 PicGo-server 来调用 sm.ms 图床的上传 api，而现在 sm.ms 的 api 是 V2 的，PicGo 上的 sm.ms 图床 api还是 V1 的，就调用不成功。



## 配置 PicGo



虽然 PicGo 自带的 sm.ms 图床接口不能用，好在它提供了一个插件市场，里面有各路大神提供的插件，我用的是一个叫 `web-uploader` 的插件，不止这个，还有其他的一些插件比如 `smms-user` 都是可以的，只要配置好上传的接口就行



![custom](https://i.loli.net/2020/03/27/YG4w2lumSTtjNWH.jpg)



下面是我的这个插件对 sm.ms 的上传接口的配置，其中 API 地址和 POST 参数名可以在 sm.ms 官网找到，返回的是一串 JSON 字符串，所以我们只要 url 链接就行了，请求头是关键，我们要将代表自己身份的 Authorization 头一起提交，否则没有认证就不知道我们是谁，一个固定 ip 每天的上传量是有限的，但是认证过后就可以上传更多的图片



![config](https://i.loli.net/2020/03/27/ADJK3ToU6xQetc2.jpg)



至于 Authorization 字样可以在用户中心找到，生成一个 API token 就行了，这个 token 不能被别人知道，否则别人可以用这个 token 假冒你的身份来上传东西，api 文档链接[在这里](https://doc.sm.ms/)，熟悉 web 开发的朋友可以直接对着文档看



![token](https://i.loli.net/2020/03/27/niX1uasHt89ZfoU.jpg)



## 填坑



本来以为挺简单的，自己搭下来还是挺多坑的，前面也说了，其实整个流程下来，都是在用 PicGo 的 server 来进行操作，所以会出错的话基本都是 PicGo 的锅，一定要去看日志找出错的原因，这也是为什么开发者要写日志的原因。



### fail to fetch



这个是因为端口的原因，typora 默认是去 36677 端口上传，所以 PicGo 也要将 server 的端口设置成 36677 ，而 PicGo 默认的端口也就是 36677，出现这个错误的原因可能是因为开了多个 PicGo 程序导致 36677 端口被占用，程序只好去开一个新的端口，看看 log 就知道了（PicGo 设置 - 设置日志文件 - 打开）



![port-error](https://i.loli.net/2020/03/27/ef16RFgaz3WShst.jpg)



解决方法就是去 `PicGo 设置 - 设置 server` 将端口号改成 36677，保存，再关闭所有的 PicGo 程序，typora 上传的时候会自动开启 PicGo 程序，所以我们没有必要提前将 PicGo 打开



### { ‘success’ : false }



这个错误我遇到很多次，莫名其妙的就是上传失败，看了网上的解决方案说是因为上传了两张相同的图片，并不是，看下 log，明明已经在 PicGo 这里上传 success 了，但返回的 markdown 链接竟然是 undefined，偶尔会出现这个问题，大部分情况下都是能正常上传的，所以就没有去管了，现在还没找到解决方法，感觉应该是 sm.ms 那边出的问题



![undefined](https://i.loli.net/2020/03/27/iak9bfQnDUZwKz2.jpg)



> 更新：在给作者提出 issue 之后确定了应该是 web-uploader 这个插件出的问题



## reference



https://zhuanlan.zhihu.com/p/114175770