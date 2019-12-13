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
---



## preface 



这是 DVWA 靶场练习系列的第三篇，这次的内容是 JavaScript 安全，相对来说比较简单



## 低级



![desc.jpg](https://i.loli.net/2019/12/12/UY95JjSgZ4lFMAD.jpg)



题目要求就是在表单中上传 `success` 字样就成功了，但是直接修改 input 框中内容的话是没用的，显示 `invalid token` ，看看源代码，在输入框前面有一个隐藏的 input 框，内容就是 token，这种一般是用来防止 CSRF 攻击的



![form.jpg](https://i.loli.net/2019/12/12/WK7SRQXjkVJ56O3.jpg)



下面有一段 js 代码就是用来生成 token 的，因此，只要计算出生成的 token 然后发送给后端就可以了，代码中会获取输入框中的值，然后进行一定加密后生成 token



![encryption.jpg](https://i.loli.net/2019/12/12/TPKYDnhoZ2UMCOd.jpg)



所以我们在输入框中填入 `success` 后直接在控制台中调用 `generate_token()` 函数就能够拿到 token，然后传给后端即可



![low-welldone.jpg](https://i.loli.net/2019/12/12/U6Tey51Wr94HaPo.jpg)





## 中级



中级的 JavaScript 文件被放到了网页外面，同样直接打开，分析一波







## 高级



## 不可能

