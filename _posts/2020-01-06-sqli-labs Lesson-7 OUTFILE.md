---
layout: post
title: sqli-labs Lesson-7 outfile
subtitle: 
date: 2020-01-06
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - sql
    - sql injection
    - security
---



## preface

这篇文章是 sqli-labs 的第七关和第八关的题解，首先我要说的是，这几关用之前双重注入的方法都可以爆出来，但是这两关要求我们用其他的方法，所以我们就用另一种方法来做一下



## 开始做题



开始的时候先试探一下能不能注入，输入 1 的时候正常，显示让我们用 `outfile`

![outfile](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\1578476350164.png)



输入单引号报错，应该有注入点，看报错信息应该是单引号周围包了两个括号，将他闭合即可，用 order by 查询字段数为 3

![column-num](https://i.loli.net/2020/01/08/mtP4YHZCU2y5DWn.png)



这题肯定也没有信息回显，所以用之前的 group by 报错方法可以爆出所有信息，但是题目要求的是 outfile ，所以我就去学习了一波 outfile 的知识，也就是将表的数据导出到本地的文件中，一般是用 

```sql
select xx into outfile 'path-to-file'
```

用 into outfile 的话需要 root 权限，并且导出的文件要写绝对路径。



这得看我们有没有权限写东西进去，毕竟是服务器，真实环境中肯定是要限制权限的，不然谁都乱写服务器还不炸？而权限的问题和 MySQL 中的一个配置 `secure_file_priv` 有关，我们来看一下靶机的这个配置是怎样的

```sql
MariaDB [security]> show variables like '%secure%';                                       +------------------+-------+                                                             | Variable_name    | Value |                                                             +------------------+-------+                                                             | secure_auth      | ON    |                                                             | secure_file_priv |       |                                                             | secure_timestamp | NO    |                                                             +------------------+-------+   
```

如果参数是空的代表没有权限设置，哪里都可以写，如果是 `null` 代表禁止导入和导出操作，如果是个特定的目录名则表明只允许向这个目录进行导入导出，这个目录一定要存在，因为 MySQL 不会自动创建它



那我们来看一下我们是不是拥有 root 权限

```sql
1' union select count(*) from mysql.user 
```













