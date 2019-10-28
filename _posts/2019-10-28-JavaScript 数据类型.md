---
layout:     post
title:      JavaScript 数据类型
subtitle:   Review notes for the tutorial of Micheal Liao
date:       2019-10-28
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - cpp
---



## preface 



本篇为廖雪峰官网 JavaScript 教程复习笔记，记下一些基本知识和 API 供自己复习用



## Number



* 整数：-1 0 1  
* 浮点数：2.33
* 科学计数法：1.23e5
* NaN：Not a Number ，无法计算时用 NaN 表示
* Infinity：无穷大，超过了 JavaScript 的最大表示值时用 Infinity 表示
* 进制数：0xffff



### Methods



* 四则运算



## String



* 'hi'
* "hi"



### Methods



TODO



## Bool



* true
* false



### Methods



* ||
* &&
* !



### Attention



* JavaScript 允许对任意数据类型作比较，用 `==` 比较会转化数据再比较，用 `===` 就不会，因此千万不要使用 `==` 进行比较，始终使用 `===` 进行比较！

  ```javascript
  false == 0; // true
  false === 0; // false
  ```

* NaN 与其他所有值都不相等，包括自己，只能用 `isNaN()` 进行判断

  ```javascript
  NaN === NaN; // false
  isNaN(NaN); // true
  ```



## null & undefined



* null 表示空值，相当于 python 中的 None
* undefined 表示值没有被定义
* 大多数情况下还是应该用 null



## Array



* var  arr = [1, 2, 3.14, 'Hello', null, true];
* var arr = new Array(1, 2, 3)



### Methods



* index



## Object



```js
var person = {
    name: 'Bob',
    age: 20,
    tags: ['js', 'web', 'mobile'],
    city: 'Beijing',
    hasCar: true,
    zipcode: null
};
```



### Methods



* key.value  e.g.  **person.name; // 'Bob'**



