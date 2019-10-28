---
layout:     post
title:      JavaScript 数据类型
subtitle:   Review notes for the tutorial of Micheal Liao
date:       2019-10-28
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - JavaScript
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

* 多行字符串

  ```js
  `这是一个
  多行
  字符串`;
  ```

* '\x41' ：用 `\x##` 十六进制形式表示 ASCII 码

* '\u4e2d\u6587' ：用  `\u####` 表示一个 Unicode 字符



### Methods



* length

* toUpperCase()

* toLowerCase()

* indexOf()  //搜索指定字符串出现的位置

  ```js
  var s = 'hello, world';
  s.indexOf('world'); // 返回7
  s.indexOf('World'); // 没有找到指定的子串，返回-1
  ```

* substring() // 返回指定索引区间的子串

  ```js
  var s = 'hello, world'
  s.substring(0, 5); // 从索引0开始到5（不包括5），返回'hello'
  s.substring(7); // 从索引7开始到结束，返回'world'
  ```

* 拼接字符串可以用 `+` 号，有很多变量要拼接的话用模板字符串（表示方法和多行字符串一样）自动替换

  ```js
  var name = '小明';
  var age = 20;
  var message = '你好, ' + name + ', 你今年' + age + '岁了!';
  alert(message);
  var message = `你好, ${name}, 你今年${age}岁了!`;
  alert(message);
  ```



## Bool



* true
* false



### Methods



* 或 ||
* 且 &&
* 非 !



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

* length

* indexOf( ) 

  ```js
  var arr = [10, 20, '30', 'xyz'];
  arr.indexOf(10); // 元素10的索引为0
  arr.indexOf(20); // 元素20的索引为1
  arr.indexOf(30); // 元素30没有找到，返回-1
  arr.indexOf('30'); // 元素'30'的索引为2
  ```

* slice( ) ：python 的切片，string 的 substring

* push() & pop()

* unshift() & shift() ：unshift 在 array 头部添加元素，shift 在头部删除元素

* sort()

* reverse()

* splice()

  ```js
  var arr = ['Microsoft', 'Apple', 'Yahoo', 'AOL', 'Excite', 'Oracle'];
  // 从索引2开始删除3个元素,然后再添加两个元素:
  arr.splice(2, 3, 'Google', 'Facebook'); // 返回删除的元素 ['Yahoo', 'AOL', 'Excite']
  arr; // ['Microsoft', 'Apple', 'Google', 'Facebook', 'Oracle']
  // 只删除,不添加:
  arr.splice(2, 2); // ['Google', 'Facebook']
  arr; // ['Microsoft', 'Apple', 'Oracle']
  // 只添加,不删除:
  arr.splice(2, 0, 'Google', 'Facebook'); // 返回[],因为没有删除任何元素
  arr; // ['Microsoft', 'Apple', 'Google', 'Facebook', 'Oracle']
  ```

* concat()  连接两个 array 并且返回一个新的 array 

  ```js
  var arr = ['A', 'B', 'C'];
  var added = arr.concat([1, 2, 3]);
  added; // ['A', 'B', 'C', 1, 2, 3]
  arr; // ['A', 'B', 'C']
  ```

* join() ：和 python 一样，将 array 用一定连接符连接成 string

  ```js
  var arr = ['A', 'B', 'C', 1, 2, 3];
  arr.join('-'); // 'A-B-C-1-2-3'
  ```





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



* obj.prop & obj['prop']  e.g.  person.name; // 'Bob'

  ```js
  var xiaohong = {
      name: '小红',
      'middle-school': 'No.1 Middle School'
  };
  xiaohong['middle-school']; // 'No.1 Middle School'
  xiaohong['name']; // '小红'
  xiaohong.name; // '小红'
  ```

* dynamic properties：

  ```js
  var xiaoming = {
      name: '小明'
  };
  xiaoming.age; // undefined
  xiaoming.age = 18; // 新增一个age属性
  xiaoming.age; // 18
  delete xiaoming.age; // 删除age属性
  xiaoming.age; // undefined
  delete xiaoming['name']; // 删除name属性
  xiaoming.name; // undefined
  delete xiaoming.school; // 删除一个不存在的school属性也不会报错
  ```

* in ： 判断对象有没有某种属性(可能通过继承所得)

  ```js
  var xiaoming = {
      name: '小明',
      birth: 1990,
      school: 'No.1 Middle School',
      height: 1.70,
      weight: 65,
      score: null
  };
  'name' in xiaoming; // true
  'grade' in xiaoming; // false
  'toString' in xiaoming; // true
  ```

* hasOwnProperty() ： 判断对象是否有自身的某种属性(非继承所得)

  ```js
  var xiaoming = {
      name: '小明'
  };
  xiaoming.hasOwnProperty('name'); // true
  xiaoming.hasOwnProperty('toString'); // false
  ```

