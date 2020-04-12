---
layout:     post
title:      DVWA靶机练习之JavaScript
subtitle:   
date:       2019-12-10
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - security
    - web
    - JavaScript
    - ctf
---



## preface



这是 DVWA 靶场练习系列的第四篇，这次的内容是 JavaScript 安全，相对来说比较简单



## 低级



![desc.jpg](https://i.loli.net/2019/12/12/UY95JjSgZ4lFMAD.jpg)



题目要求就是在表单中上传 `success` 字样就成功了，但是直接修改 input 框中内容的话是没用的，显示 `invalid token` ，看看源代码，在输入框前面有一个隐藏的 input 框，内容就是 token，这种一般是用来防止 CSRF 攻击的



![form.jpg](https://i.loli.net/2019/12/12/WK7SRQXjkVJ56O3.jpg)



下面有一段 js 代码就是用来生成 token 的，因此，只要计算出生成的 token 然后发送给后端就可以了，代码中会获取输入框中的值，然后进行一定加密后生成 token



![encryption.jpg](https://i.loli.net/2019/12/12/TPKYDnhoZ2UMCOd.jpg)



所以我们在输入框中填入 `success` 后直接在控制台中调用 `generate_token()` 函数就能够拿到 token，然后传给后端即可



![low-welldone.jpg](https://i.loli.net/2019/12/12/U6Tey51Wr94HaPo.jpg)





## 中级



中级的 JavaScript 文件被放到了网页外面，同样直接打开，分析一波逻辑，也不难，直接就是在输入框内容的两边各加一个 `XX` 然后再逆序过来就是 token 的值，因此，token 就是 `XXsseccusXX`



![js-logic.jpg](https://i.loli.net/2019/12/13/ie9WgYIpLcw5xNd.jpg)



## 高级



高级的例题也是将 js 放到了网页外面，然而打开之后，特别长一串，一看就是混淆过的



![obfuscate.jpg](https://i.loli.net/2019/12/13/EIvOQKRCDtkq82o.jpg)



然后就找去混淆的网站，发现没有啥能够完全去混淆的网站，都是去除了很多有用的信息，因此就找了网上的题解，找到了一个去 js 混淆的[网站](http://deobfuscatejavascript.com/)，将代码贴到网站上去混淆，这次终于可以了，下面是核心的代码



![core-code.jpg](https://i.loli.net/2019/12/13/M3bFSmUtc5ekOT9.jpg)



分析一下逻辑，有点奇怪，代码先是对 phrase 框进行清空，然后调用 `token_part_1("ABCD", 44)` 函数，这两个参数都没什么卵用，因为这个函数并没有用到这两个参数，所以只是将 phrase 框中的内容翻转过来而已，按照标准答案，phrase 里面是 `success` ，但是之前明明已经对 phrase 进行清空了，这里没想通。



之后再调用 `token_part_2("XX")` 这个函数，也就是和上一步得到的 token 进行 sha256 加密，然后再点击按钮的时候调用 `token_part_3()` 这个函数，继续进行 sha256 加密，得到最终的 token 



![high-solution.jpg](https://i.loli.net/2019/12/13/dtP2e8ZGf9KMmXJ.jpg)



后来想了一下，反正题目的意思就是要在 phrase 中填入 `success` 并且发送正确的 token ，而且生成 token 的顺序也告诉我们了，那就不要就纠结太多，知道该怎么做就行了



## 不可能



![impossible.jpg](https://i.loli.net/2019/12/13/8OrWaFRgemolE6H.jpg)



永远不要相信用户的输入，所以只要有输入的地方就有漏洞，没有绝对的安全23333