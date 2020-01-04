---
layout: post
title: sqli-labs Lesson-2 数字型注入
subtitle: 
date: 2020-01-03
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - sql
    - sql injection
    - security
---



## preface

这篇文章是 sqli-labs 的第二关题解，方法和第一关差不多，所以这里就只记录一下关键的点



按照惯例，首先敲一个 `'` 进去看会不会报错，然后报错了

![error](https://i.loli.net/2020/01/04/Xm4GEcOPjMTsDdy.png)



根据这个信息我们就知道了这可能存在数字型注入，为什么呢，我们看后面一部分报错信息

```
near '' LIMIT 0,1' at line 1
```

我们输入了一个单引号，这个单引号的周围并没有其他的引号，因此可以猜测 sql 语句为

```sql
select xx from table where id=$id limit 0, 1
```



第一题字符型注入的报错信息是下面这样的，引号外面包了一组单引号，注意这些差别

```
near '''' LIMIT 0,1' at line 1
```



然后就是按部就班，该爆什么就爆什么，跟字符型注入相比只是不用加引号了而已，而且 union 查询如果想要让前面一个查询结果为空的话可以直接填个 `-1` ，或者 `1 and 1=2` 



下面爆一个数据库看看

![database](https://i.loli.net/2020/01/04/BrKLlFv5VNEPwf1.png)



其余的步骤都大同小异，看我[第一关的过关步骤](https://szukevin.site/2020/01/03/sqli-labs-Lesson-1-%E5%AD%97%E7%AC%A6%E5%9E%8B%E6%B3%A8%E5%85%A5/)就行了