---
layout: post
title: win10平台nginx+php+mysql填坑之路
subtitle: 一次作死的瞎搞
date: 2019-11-14
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - php
    - sql
    - nginx
---



## preface 



前面不是配置好了 php 和 MySQL 的环境了嘛，就想着干脆再搞个服务器好了，以前一直在 Windows 上用的是 apache ，用着没什么事，但是越来越多的网站选择用 nginx 了，所以我就试了一下在 Windows 上配置 nginx 的环境，给大家把坑都填上了



## 安装软件



这个大家自己去官网下载就行了，我假设大家已经下载完，并且已经把 php 和 mysql 加入环境变量了



我的安装路径如下：

> php:  F:\php-7.2.24-Win32-VC15-x64
>
> mysql: F:\MySQL
>
> nginx: F:\nginx-1.16.1



## 进行配置



首先进行 php 和 MySQL 的连接配置，这个还是比较简单的，只需要在 `php.ini` 文件里找到下面这条，并且取消注释就行了，[之前的文章](https://yarkable.github.io/2019/11/06/PhpStorm%E9%85%8D%E7%BD%AEXdebug%E6%89%93%E9%80%A0%E8%88%92%E9%80%82php%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83/)里面也有提到

```ini
;extension=mysqli
```



然后将 `my-default.ini` 拷贝一份，改名为 `my.ini` ，在里面写入如下内容：

```ini
[mysql]
# 设置mysql客户端默认字符集
default-character-set=utf8 
[mysqld]
#设置3306端口
port = 3306 
# 设置mysql的安装目录
basedir=F:\MySQL
# 设置mysql数据库的数据的存放目录
datadir=F:\MySQL\data
# 允许最大连接数
max_connections=200
# 服务端使用的字符集默认为8比特编码的latin1字符集
character-set-server=utf8
# 创建新表时将使用的默认存储引擎
default-storage-engine=INNODB 
```



接下去就是 nginx 的配置，直接双击 nginx.exe 就已经启动了服务，在浏览器中输入 `localhost` 就会出现 nginx 的欢迎界面



![](C:\Users\kevin\Desktop\blog\nginx.jpg)



然后我们要让他能够解释 php 文件，我们是用 php-cgi 来作为解释器的，大概就是让 php-cgi 监听 9000 端口，将 php 文件传入 php-cgi 然后将解释过的内容渲染在浏览器上。在 `nginx-1.16.1/conf/nginc.conf` 的第 43 行，加上 `index.php` 作为默认网页：



```ini
location / {
    root   html;
    index  index.html index.htm index.php;
}
```



然后我们在 `php.ini` 中进行配置，使其能和 nginx 一起使用。找到如下语句，有注释的就取消注释，值不一样的就改成下面的值：

```ini
date.timezone = Asia/Shanghai
enable_dl = On
cgi.force_redirect = 0
fastcgi.impersonate = 1
cgi.rfc2616_headers = 1
cgi.fix_pathinfo=1
```



> 先说下出问题的做法



在 `nginx.ini` 第 65 行以后取消注释，就是下面这样子

```ini
location ~ \.php$ {
    root           html;
    fastcgi_pass   127.0.0.1:9000;
    fastcgi_index  index.php;
    fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
    include        fastcgi_params;
}
```



然后我们输入下列命令让 php-cgi 监听 9000 端口(转到 php 的安装路径)

```powershell
php-cgi.exe -b 127.0.0.1:9000 -c php.ini
```



终端并没有任何输出，有点像卡住的样子，其实已经在跑了，我们可以用以下命令查看 9000 号端口的情况，如果输出 `LISTENING` 就说明 php-cgi 起作用在监听了



```powershell
netstat -ano|findstr "9000"
```



这时表面上已经完成配置了，我们在 `nginx/html` 文件夹里新建一个文件 `index.php` 用来测试：

```php
<?php
    phpinfo();
?>
```



在浏览器中输入 `localhost/index.php` ，出事了，直接就叫我下载 `index.php`了，说明 php-cgi 程序并没有正确解释，因此，我参照网上的说法又对 `nginx.conf` 进行了如下修改，将 `root` 地址改为了站点的绝对路径，`fastcgi_param` 中的 `/script` 改成了 `$document_root` ，这个就代表了上面的 root 目录。

```ini
location ~ \.php$ {
    root           F:\nginx-1.16.1\html;
    fastcgi_pass   127.0.0.1:9000;
    fastcgi_index  index.php;
    fastcgi_param  SCRIPT_FILENAME  $document_root$fastcgi_script_name;
    include        fastcgi_params;
}
```



再次运行服务，在浏览器中输入 `localhost/index.php` ，这回是新的错误，直接返回下面这段话：

```
No input file specified.
```



行吧，这总比之前好点，上网搜解决方案，告诉我的全是假的，没什么卵用，最后想起之前好像有人说过路径一定要加引号，我也试了下，没卵用，经过了多次尝试和 Google 之后，找到了正确的解决方案，下次就直接进入任务管理器杀掉 `nginx.exe` 再用命令行打开就解决了，因此，正确的配置应该是下面这样：



```ini
location ~ \.php$ {
    root           "F:/nginx-1.16.1/html";
    fastcgi_pass   127.0.0.1:9000;
    fastcgi_index  index.php;
    fastcgi_param  SCRIPT_FILENAME  $document_root$fastcgi_script_name;
    include        fastcgi_params;
}
```



## 编写 bat 脚本



下载 [RunHiddenConsole](https://redmine.lighttpd.net/attachments/660/RunHiddenConsole.zip) ，顾名思义，这个玩意会将跑程序的终端隐藏起来，像 php 和 php-cgi 等程序开启之后一直挂在那里，我们不能用那个



## reference



