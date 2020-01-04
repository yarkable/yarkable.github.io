---
layout: post
title: sqli-labs Lesson-1 字符型注入
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



这篇文章是我正式学习 sql 注入的第一篇文章，下载了 sqli-labs 靶机作为渗透环境，里面的题目很多，有几十题的 sql 注入，跟着网上的 wp 一起做了一题，然后还是学到挺多东西的，这里拿第一关来记录一下



## sql 注入概念



通俗点说，就是要获取数据库里面所有的信息，包括这张表的字段个数，字段名分别是什么，在哪个数据库中，以及所有的记录，一般来说，黑客就会想知道有关管理员表的一些信息，作为学习，我就先将 sql 注入的一般的步骤写下来



1. 判断是否可注入以及注入点的类型（字符型，数字型，布尔型）
2. 猜解表中的字段数（一般利用 order by column_id）
3. 确定显示的字段顺序（一般用 union 联合查询）
4. 获取当前的数据库（通过 MySQL 内建的 database() 函数）
5. 获取表中的字段名
6. 下载数据



## 

