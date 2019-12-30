---
layout: post
title: Linux平台nginx+php+mysql填坑之路
subtitle: 顺便再绑个域名
date: 2019-12-26
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - php
    - sql
    - nginx
    - linux
---



## preface 



这篇是 Linux 下配置 lnmp 的教程，和 win10 下配置 wamp 教程是姐妹篇，网上的教程质量参差不齐，官网又没有给出安装教程，所以就自己记录一下，下次要装的话就不用再去找网上的教程了，此篇文章环境为 Kali Linux 



## 安装软件



主要安装的还是 php ，nginx 和 MySQL ，怎么说呢，相比 Windows 还是要更加简单一点的，只不过刚开始要去搜索它们的配置文件存放在哪儿，先装 nginx

```bash
root@kali:~# sudo apt install nginx
```



再装 php，同时要装很多附加的东西

```bash
root@kali:~# sudo apt install php php-fpm php-MySQL
```



最后就是 MySQL，不过 Kali 系统上只有 mariadb 可以用，这比 MySQL更高级一点，是 MySQL 的衍生版本

```bash
root@kali:~# sudo apt install mariadb-client mariadb-server
```



这样安装就结束了，不过 Kali Linux 自带了 php 和 MySQL ，用的是 apache 服务器，所以我这里自己下载了 nginx ，不想折腾的直接用系统自带的 apache 就行了



## ATTENTION



写在前面，下面每次更改配置之后都要重启 nginx ，一定要记得重启，否则改动是不会生效的，php-fpm 也一样，改动过后都要重启



```bash
$ service nginx restart
$ service php7.3-fpm restart
```





> Windows 下用的是 php-cgi ，Linux 可以选择更强大的 php-fpm ，因为 php-fpm 是用 Linux 的 fork() 来工作的，所以 Windows 上用不了



## 进行配置



### php



首先来配置 php 的功能，单独的 php 只是个 CLI 工具，也就是在命令行中调用，而 php-fpm 可以在网络中被调用，是一个 CGI(Common Gateway Interface) 应用，它可以监听计算机端口，得到请求然后返回结果，这和 php-cgi 的功能是一样的，不过 php-fpm 是加强版，一般监听的端口为 9000，所以我们来进行配置

```bash
root@kali:~# vim /etc/php/7.3/fpm/pool.d/www.conf 
```



打开 php-fpm 的配置文件后，找到下面这行，将其注释，然后写上一行 `listen = 127.0.0.1:9000` 代表 php-fpm 监听本机的 9000 端口。这里用的是网络端口监听，上面被注释的是用 sock 协议进行监听，一般用在 nginx 和 php 装在同一台机器中的情况，通常情况下 sock 的效率更高，我们待会儿再讲，先来配置本地端口监听的 php



![php-fpm-config](https://i.loli.net/2019/12/29/8IR9l2jovmNcrpB.png)





### nginx



然后就配置我们的服务器，首先看看 nginx 目录下的文件，默认安装位置为 `/etc/nginx`

```bash
.
├── conf.d
├── fastcgi.conf
├── fastcgi_params
├── koi-utf
├── koi-win
├── mime.types
├── modules-available
├── modules-enabled
│   ├── ......
├── nginx.conf
├── proxy_params
├── scgi_params
├── sites-available
│   └── default
├── sites-enabled
│   ├── ctf.conf
│   └── default -> /etc/nginx/sites-available/default
├── snippets
│   ├── fastcgi-php.conf
│   └── snakeoil.conf
├── uwsgi_params
└── win-utf

```



刚刚下载完的时候，会在 `/vaw/www/` 文件夹中新建一个 `html` 文件夹，里面有一个 index.html ，这是 nginx 的默认首页，我们可以用我们网站的数据替换掉 `html` 文件夹中的内容，也可以新建一个文件夹，重新配置一个站点，这里我们就重新配置一下



首先，在 `/var/www/` 中新建一个目录，由于我是用来搭建 ctf 靶机，我的目录就叫 `upload-labs` ,这就是我们网站的根目录，然后把网站的数据粘贴进文件夹，这是我的文件夹的结构

```bash
/var/www/upload-labs/
├── index.php
├── css
├── doc
├── docker
├── img
├── js
├── Pass-01
├── Pass-02
├── ......
└── upload
```



然后就去 nginx 的配置文件中进行配置，我们先看 `nginx.conf` 这个文件，这里面是一个总的 nginx 配置，有 http ，events ，mail 等几个配置块

```
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;
events {
	...
}
http {
	##
	# Virtual Host Configs
	##
	include /etc/nginx/conf.d/*.conf;
	include /etc/nginx/sites-enabled/*;
}
mail {
	...
}
```



主要是 http 这个块，可以看到，最后他加了两行 `include` ，说明我们可以将 http 的配置写在其他的文件中，最后引用进来就行了，所以为了保持原生的配置文件，我们就去 `sites-enabled` 这个文件夹新建一个名叫 `ctf.conf` 的配置文件（里面有个默认的配置文件，里面有很多模板配置，我们将自己需要的复制一下就行了）

```
server {
	listen 80;
	listen [::]:80;

	server_name localhost;

	root /var/www/upload-labs;
	index index.html index.php;

    location ~ [^/]\.php(/|$)
    {
        #fastcgi_pass unix:/run/php/php7.3-fpm.sock;
        fastcgi_pass 127.0.0.1:9000;
        fastcgi_index index.php;
        fastcgi_param  SCRIPT_FILENAME  $document_root$fastcgi_script_name;
        include fastcgi_params;
    }
}
```



下面说下这些配置都是啥意思

| 配置        | 意义                                                         |
| ----------- | ------------------------------------------------------------ |
| listen      | 表示监听的端口号                                             |
| server_name | 网站的域名，没有的话直接填 localhost ，有的话可以填自己购买的域名 |
| root        | 网站根目录的存储位置                                         |
| index       | 默认的首页的名称                                             |
| location    | 处于某个位置时应该执行的动作，这里通过 nginx  的正则表达式，如果 nginx 接收到 php 后缀的请求就交给监听 127.0.0.1:9000 处的 php-fpm 去处理（否则就直接下载该 php 文件） |



这里的配置大概说一下，先能简单明白就好，至于反向代理，负载均衡之后再讲，这里主要是 location 需要理解一下，因为 nginx 是不能解释 php 内容的，初始状态下如果它遇到了 php 后缀的文件，就会当成是普通的文件，然后直接让我们下载，不会在网页中渲染，所以我们要让 fastcgi 程序来处理 php 文件，也就是说对于 fastcgi 来说（这里是 php-fpm），nginx 相当于客户端，fastcgi 相当于服务器

下面这行就是调用 fastcgi服务，由于我们配置的是端口监听，所以这里是监听本地端口 9000 ，这时默认的端口

```
fastcgi_pass 127.0.0.1:9000;
```

上面这种的好处呢就是可以不监听本地端口，可以监听网络上的装了 fastcgi 应用的计算机的端口。但如果我们只想用本地的 php 服务的话完全可以用 sock 协议，也就是上面被注释的那行，这个效率是更高的，网上的一些教程只是说注释了它，没有说为什么注释，其实完全不用注释，只要 `/etc/php/7.3/fpm/pool.d/www.conf` 里面监听的内容和这里是一样的就可以了，我已经亲身试过了，所以网上的教程还是不要全信，看完之后要有点自己的思考

```
fastcgi_pass unix:/run/php/php7.3-fpm.sock;
```



下面的两行，都是跟 `fastcgi-param` 有关的，也就是 nginx 与 fastcgi 服务器交互的一些参数，我们先来看看默认情况下是怎样的

```
fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
include        fastcgi_params;
```



第一行，表示的是 nginx 传输文件给  fastcgi 时的具体文件名，fast-cgi 会去网站根目录的 `/script` 中找这个文件，这有什么不好的地方呢，如果 php 文件不在 /script 中，fastcgi 就找不到他了，因此我们要将这个参数 `/script` 修改成网站的根目录（也就是 root 的位置），然后 nginx 官方也发现这样太麻烦了，就提供了一个参数 `$document_root` 指向当前的网站根目录，所以我们可以这样修改一下

```
fastcgi_param  SCRIPT_FILENAME  $document_root$fastcgi_script_name;
```



第二行，就是将 nginx 目录下的 `fastcgi-parms` 给引进来了，也就是说在 `fastcgi_params` 中没有对 `SCRIPT_FILENAME` 做定义，所以要先定义，然后再将其他的变量给引入，说个好消息，在新版的 nginx 中其实已经对这些东西做好了定义，就在 `fastcgi.conf` 中，所以我们可以直接将这两行给替换掉，直接用下面的这行语句代替会更加方便！

```
include        fastcgi.conf;
```



到这一步其实就已经可以了，我们已经可以在浏览器中访问我们的网站了，下一步配置一下数据库

![upload-labs](https://i.loli.net/2019/12/30/2V3j7IBZSQq5PrK.png)



### MySQL









## 编写 bat 脚本



上面这些命令太麻烦了，在 Windows 上敲命令真是件不愉快的事情，因此我们可以将这些命令集合成一个脚本，每次只要运行一下脚本就运行了所有命令！



下载 [RunHiddenConsole](https://redmine.lighttpd.net/attachments/660/RunHiddenConsole.zip) ，顾名思义，这个玩意会将跑程序的终端隐藏起来，像 php 和 php-cgi 等程序开启之后一直挂在那里，我们不能用那个终端干其他事，就很烦，然后我们可以在 windows 上用这个将他们隐藏起来。下载 RunHiddenConsole 后将它解压在 nginx 的安装文件中



![nginx2.jpg](https://i.loli.net/2019/11/15/aRQ3Ui7CfST2ykr.jpg)



在当前目录编写启动脚本 `start_nginx.bat` ：



```shell
@echo off
REM Windows 下无效
REM set PHP_FCGI_CHILDREN=5

REM 每个进程处理的最大请求数，或设置为 Windows 环境变量
set PHP_FCGI_MAX_REQUESTS=1000

echo Starting PHP FastCGI...
RunHiddenConsole F:\php-7.2.24-Win32-VC15-x64\php-cgi.exe -b 127.0.0.1:9000 -c F:\php-7.2.24-Win32-VC15-x64\php.ini

echo Starting nginx...
RunHiddenConsole F:\nginx-1.16.1\nginx.exe
```



再编写一个 `kill_nginx.bat` ：



```shell
@echo off
echo Stopping nginx...
taskkill /F /IM nginx.exe > nul
echo Stopping PHP FastCGI...
taskkill /F /IM php-cgi.exe > nul
exit
```



以后双击两下 `start_nginx.bat` 就打开了服务，双击 `kill_nginx.bat` 就将服务给关闭了，很方便，不用自己再去开那么多命令行窗口手动敲命令了！



## reference



https://mobilesite.github.io/2017/03/26/config-and-use-of-php-nginx-MySQL/

https://my.oschina.net/kenshiro/blog/187926

https://xuchen.wang/index.php/archives/nginxphp.html