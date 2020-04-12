---
layout:     post
title:      DVWA靶机练习之File Inclusion
subtitle:   
date:       2019-12-06
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - security
    - web
    - php
    - ctf
---



## preface



这是 DVWA 靶场练习系列的第五篇，这次的内容是文件包含漏洞，即服务器通过 php 的特性（函数）去包含任意文件时，由于要包含的这个文件来源过滤不严，从而可以去包含一个恶意文件，而我们可以构造这个恶意文件来达到邪恶的目的。



>  涉及到的危险函数：include(),require()和include_once(),require_once()



并且被包含的文件都是被当作 php 文件来执行的，不管文件的后缀是什么，只要内容是 php 格式的话，就可以被执行，如果内容不是 php，则会将内容直接打印出来，常常和文件上传漏洞相配合使用



## low



给出三个 php 文件，利用 GET 的形式获取文件，然后将文件包含

![inclusion](https://i.loli.net/2020/04/12/3pcIFZby1rsz6mj.png)

如果我们填上一个不存在的文件，看到服务器就会报错，直接就把根目录给搞出来了

![error](https://i.loli.net/2020/04/12/a6xIo9GCiMRr5OV.png)



那么我们就可以直接读取服务器上的任意文件了，先读个 ini 文件看看

![ini-read](https://i.loli.net/2020/04/12/7Iwvzqcb9xS1rYA.png)



反正只要知道服务器上文件的位置，就可以进行读取了，一般如果知道了某个 webshell 的位置，用文件包含漏洞可以直接连上网站后台



本地文件包含：

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=../../php.ini
```

远程文件包含：

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=http://localhost/dvwa/php.ini
```



## medium



中级的代码按照 low 级别的方法就不行了，直接会报错，看看源代码发现是将路径和 http 协议进行了字符串替换，其实这个还挺简单的，那就直接双写就可以绕过了

![medium](https://i.loli.net/2020/04/12/magkwWq1JoltVXT.png)



本地文件包含：

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=..././..././php.ini
```

远程文件包含：

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=hthttp://tp://localhost/dvwa/php.ini
```

或者直接用绝对路径(low 级别已经知道了服务器绝对路径)：

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=F:\nginx-1.16.1\html\dvwa\php.ini
```



## high



high 级别输入一个地址就会返回 `ERROR: File not found!`，没什么思路，查看源代码，只有 page 参数 为 `include.php` 或者以 `file` 开头的文件才能够绕过，否则都会报错说文件不存在



![high](https://i.loli.net/2020/04/12/HlvMbi4serKTNRx.png)

那其实就可以利用 `file` 协议进行本地文件包含(file:///)，当我们在浏览器中打开一个本地文件时用的就是 `file` 协议

```txt
http://localhost/dvwa/vulnerabilities/fi/?page=file:///F:\nginx-1.16.1\html\dvwa\php.ini
```



