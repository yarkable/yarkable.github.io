---
layout: post
title: upload-labs文件上传漏洞
subtitle: 
date: 2020-03-31
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - php
    - security
    - ctf
---



## preface



基于 upload-labs 靶机进行文件上传漏洞学习，网上通关教程很多，这里我只记录下我觉得重要的和容易忘的知识点，感谢 [c0ny](https://github.com/c0ny1/upload-labs) 大佬



需要用到的工具：

* Burp Suite
* [蚁剑](https://github.com/AntSwordProject/antSword/releases)



## 上传漏洞思维导图



来源于 [c0ny](https://github.com/c0ny1/upload-labs) 大佬的 Github 仓库，总结得挺到位的，所以在这里贴一下（侵删），文件上传漏洞有两个大类，一个是服务器端用代码验证文件时没有过滤完全，另一大类就是利用服务器的解析漏洞上传 webshell：Apache，IIS，Nginx 都存在着解析漏洞。



![image.png](https://i.loli.net/2020/04/01/CLQX84GDMltJ63j.png)



![](https://i.loli.net/2020/04/01/3IzLb8A1ESirsH2.png)

php 可解析列表

```text
array(
    ".php",".php5",".php4",".php3",".php2","php1",
    ".html",".htm",".phtml",".pht",".pHp",".pHp5",".pHp4",".pHp3",
    ".pHp2","pHp1",".Html",".Htm",".pHtml",".jsp",".jspa",".jspx",
    ".jsw",".jsv",".jspf",".jtml",".jSp",".jSpx",".jSpa",".jSw",
    ".jSv",".jSpf",".jHtml",".asp",".aspx",".asa",".asax",".ascx",
    ".ashx",".asmx",".cer",".aSp",".aSpx",".aSa",".aSax",".aScx",
    ".aShx",".aSmx",".cEr",".sWf",".swf"
);
```

php6 也可以 



## 服务器解析漏洞



先记一下一些常用到的服务器解析漏洞，不怎么用到的就不记了，以后做到相关题目再来更新，而且网上关于服务器解析漏洞的博客内容竟然全都是？？？一样的？？？真的服了，抄来抄去…



### Windows环境



* Windows 保存文件不允许后缀的最后有 `.` 或者空格，如果服务器在 Windows 上的话，上传一个 webshell 名叫 `shell.php(空格)` 或者 `shell.php.` ，服务器会自动将其重命名为 `shell.php` ，这样就上传了一个 shell
* Windows 文件后面加个 `:: $DATA` 的话， `:: $DATA` 后面的内容会被当成流文件处理，不会被当成后缀，因此 `shell.php:: $DATA` 上传到 Windows 服务器就是 `shell.php`



### 服务器类型



### Apache



* Apache 在解析文件时按照后缀从右往左进行解析，如果上传了一个 `shell.php.abc.fuck` 上去的话，Apache 解析不了后面两个后缀而能够解析 php 的话，它就会将文件当作 php 文件来解析。

* `.htaccess` 配置文件可以干很多事，在上传漏洞中经常用来修改 MIME 类型，在服务器上传一个 `.haccess` 文件里面写上如下内容就可以将其所在目录以及子目录中所有的文件都当作 php 文件来解析，即使给 webshell 命名为 `.jpg` 后缀也照样当成 php 解析

  ```ini
  SetHandlerapplication/x-httpd-php
  ```

* Apache 配置文件中会有 `.+.ph(p[345]?|t|tml)` 此类的正则表达式，被当成 php 程序执行的文件名要符合正则表达式，也就是说 `php3，php4，php5，pht，phtml` 也是可以被解析的（Nginx 不行）



### Nginx



* Nginx 文件漏洞是从左到右进行解析，既可绕过对后缀名的限制，又可上传木马文件。另外，如果服务器只检查文件名的第一个后缀，那么满足验证要求即可成功上传。但是对于文件来讲，只有最后一层的后缀才是有效的，如``shell.jpg.php`，那么真正的后缀应该是 php 文件，根据这个我们可绕过相关验证进行上传



## 检测漏洞流程



### 客户端 or 服务端？



首先看到一个上传点，上传一个奇怪后缀的文件如果很快返回结果的话，那么检验就是在客户端，基本就是 Javascript 代码检验，如果过了一会儿才返回结果，说明是在服务器端进行的检验，因为发送请求接受请求需要时间。



如果是在客户端的话那么可以禁用浏览器的 JavaScript 功能，也可以将 webshell 的后缀改成能通过检验的后缀，然后用 Burp 抓包，对后缀进行修改，一般常用的是后面这种方法，**见第一关**。不过关于 JavaScript 限制的问题都可以用一个更骚的方法解决，那就是用 Burp 拦截服务器返回的 Response，将 Response 中的 js 代码篡改之后再让浏览器渲染。



比如第一关的 js 限制了只能上传指定后缀的文件，其中不包括 php，那我们就可以在返回 Response 的时候将这段代码修改，添加上 php 后缀。

![before](https://i.loli.net/2020/04/07/uObqrQZozlFdayC.png)



默认情况下 Burp 是没有开启拦截 Response 的，只会拦截 Request，所以要先修改一下 Option，勾选住  `Intercept Server Responses`，这样就可以同时拦截到 Request 和 Response 了



![burp-setting](https://i.loli.net/2020/04/07/6byGYtT9FdSxa2E.png)



之后刷新一下页面，不出意外应该会被 Burp 拦截，然后将 Request 放行，修改 Response，将 php 后缀添加上去，然后点击 forward 放行



![modify-response](https://i.loli.net/2020/04/07/kVLP3Y4bZ9Bh8lA.png)



之后再浏览器中渲染出来的 js 代码就是修改过后的，这时我们就可以突破限制上传 php 脚本了



![](https://i.loli.net/2020/04/07/eJA4hqazxjKHGMt.png)



---



服务端进行检验的话就有很多种情况，下面写的都是针对服务端检查。我觉得第一步就是要判断服务器的类型(IIS? Apache? Nginx?)以及服务器所处的环境(Windows? Linux?)，这个一般可以通过浏览器地址栏路由到网站的一个不存在的地点，然后就会报 404 错，这时就可以知道服务器的一些信息，或者直接在浏览器用 F12 查看 HTTP 响应头也是可以的。



![404notfound](https://i.loli.net/2020/04/04/6wkX9RBLSNbaqJ5.png)



### 检查后缀？



#### 白名单 or 黑名单？



白名单和黑名单一般都是程序员定义的一个数组 array，白名单就是限制了只有带数组中的后缀的文件才能被上传到服务器，黑名单就是带有出现在数组中的后缀的文件都不能上传，一个典型的黑名单如下 (不全，php6，PHP 等都没有出现在黑名单中 ：)

```php
array(
    ".php",".php5",".php4",".php3",".php2","php1",
    ".html",".htm",".phtml",".pht",".pHp",".pHp5",".pHp4",".pHp3",
    ".pHp2","pHp1",".Html",".Htm",".pHtml",".jsp",".jspa",".jspx",
    ".jsw",".jsv",".jspf",".jtml",".jSp",".jSpx",".jSpa",".jSw",
    ".jSv",".jSpf",".jHtml",".asp",".aspx",".asa",".asax",".ascx",
    ".ashx",".asmx",".cer",".aSp",".aSpx",".aSa",".aSax",".aScx",
    ".aShx",".aSmx",".cEr",".sWf",".swf"
);
```



建议过滤后缀的话用白名单，因为黑名单总有办法可以绕过，下面就说说





