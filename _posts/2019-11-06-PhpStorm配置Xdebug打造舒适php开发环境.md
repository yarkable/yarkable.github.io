---
layout:     post
title:      PhpStorm配置Xdebug打造舒适php开发环境
subtitle:   php是全世界最好的语言(误
date:       2019-11-06
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
        - php
---



## preface



朋友们，是这样的，没错，我又开始学 php 了，真香啊，之前是谁说过这辈子不再学 php 的来着？一定不是我！



说一下我的学习经历吧，大二上学期上了我们学校一个老师的 JavaScript 网页特效课，虽然他没有讲什么，但是我自己了解到了一些 web 的知识，就瞎搞了一下子，他说大二下学期会开 php 课，我可高兴了，因为我是自动化专业的，平时专业开的课都是些跟电路有关的，实在没什么兴趣，因此能接触到这种编程课实在是很开心，然后，放假就抱着本 php 和 MySQL 的书看着，用过 wamp 集成环境，也自己单独搭建过 apache + MySQL + php 的环境。最后因为那节 php 课只有三个人选就被迫退课了，据说是因为上学期的 JavaScript 太恶心了hhhhh



好吧，反正那时就没人教我，然后自己也觉得 php 已经过时了，现在 node.js , python 火的不得了，就发誓再也不学 php 了，转去了 python ，现在由于一些特殊原因，又回来了，这次，有了 C++ 的基础，学起 php 的语法就感觉是照搬 C++ 的，因此，三个小时就看完了菜鸟教程的 php 内容。



bb 得也差不多了，今天就先来配置环境吧，以前用的是 sublime Text，现在玩玩 JetBrains 家的 PhpStorm，反正学生白嫖不要钱。



## 安装 PhpStorm



直接上[官网](https://www.jetbrains.com/phpstorm/)，下载，安装，完事，用学生账号注册的话是免费使用 professional 版本的，JetBrains 家的东西就是体积大了点，香还是挺香的。



![image.png](https://i.loli.net/2019/11/06/NBWXfDZTHUVRuak.png)



## 安装 php



我们下载下来 PhpStorm 之后还不能直接运行 php 文件，会报错，因为我们没有 php 解释器，这就跟 pyCharm 一样，只是个壳子，因此我们还得自己下载 php 解释器，直接上[官网](https://www.php.net/)，自己选一个版本，我下载的是 7.2 版本的 ，注意这是有两个版本的(线程安全和非线程安全)，最好下载线程安全的版本



![php](https://i.loli.net/2019/11/06/kITc7Xi5E3Brp8f.png)



## 安装 Xdebug



下载完 php 先放一放，我们再来下载 Xdebug，注意注意，这个和 php 的版本也要对应，要和自己电脑的位数相对应，而且要看清楚是否是线程安全的版本，TS 就代表线程安全(Thread Safe)，没有的就是 NTS 非线程安全(None Thread Safe)，我就因为下载了非线程安全版本就报错了，找了好久的错！



![xdebug](https://i.loli.net/2019/11/06/TkUc5B2hwvHljON.png)



## 配置 php.ini



好的，三样东西都齐活了，就下去就让他们组装在一起，首先，我们对 php 进行配置，时隔这么久，我都忘记怎么配置的了，只能去网上找教程了，在 php 安装目录里面有个 `php.ini-development` 文件，我们拷贝一份，重命名为 `php.ini` ，之后的配置就在 `php.ini` 中修改



首先，在 ini 文件中找到 `extension_dir` 这一行，取消前面的 `;` 注释，再修改 php 拓展路径为 php 安装路径中的 `ext` 文件夹，否则默认是在 `C:\PHP\ext` 中



```
 extension_dir = "Your PHP Dir\ext"
```



然后我们再将下载好的 Xdebug 插件移动到上面的 ext 文件夹中（这里我有两个，是因为我下错了一个版本==）



![xdebug.dll](https://i.loli.net/2019/11/06/Kv28xRGgSlWsafe.png)



之后在 php.ini 中添加 Xdebug 的扩展，就在配置文件的最后加上下面的几行（这里是我的配置，路径因人而异哈）



```
[xdebug]
zend_extension="F:\php-7.2.24-Win32-VC15-x64\ext\php_xdebug-2.8.0-7.2-vc15-x86_64.dll"
xdebug.remote_enable=1
xdebug.remote_port=9000
xdebug.idekey=PHPSTORM
```



**更新**



 php 要使能 MySQL 的话还要在 ini 文件中找到下面几句话，取消注释

```sql
extension=mysqli
date.timezone = Asia/Shanghai
```



## 配置 PhpStorm



最后一项，在 PhpStorm 中配置 php 解释器，按下 `ctrl+alt+s` 打开设置面板，选择 `Languages & Frameworks` 中的 `PHP` 选项卡，然后选择我们的 php 版本，并且在系统路径中寻找到我们的 php 解释器



![phpstorm](https://i.loli.net/2019/11/06/7EVgkyHa4w6QWib.png)



如果之前的 Xdebug 配置正确的话，是可以直接找到 debugger 的信息的，如果没装对的话就得看看是不是哪里出了问题，Xdebug 官网有个[工具](https://xdebug.org/wizard)可以智能检测到我们缺少哪些东西并给出建议，只需要把我们的 phpinfo() 函数里面输出的信息或者在终端输入 `php -i` 之后输出的内容复制进去，就能知道我们接下来需要做什么，就像下面这样子



![wizard](https://i.loli.net/2019/11/06/wmsunI69tdL51Dx.png)



到这一步的话就可以了，我们随便写个 php 脚本，然后按下运行，开冲！发现这种牛逼的 IDE 的智能补全真的到位，一直按着 tab 就完事了，都不用动脑子。



![php-script](https://i.loli.net/2019/11/06/A3D7Yia4mNXken8.png)

