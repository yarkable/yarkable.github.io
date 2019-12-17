---
layout:     post
title:      在katacoda上学习docker
subtitle:   docker是个好东西！
date:       2019-12-14
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - docker
---



## praface 



早就知道 docker 很重要了，但是网上很多 docker 教程都讲的不清楚，doker 是啥，docker 能干啥，为什么要 docker，怎么用 docker，偶然的机会才知道了[这个网站](https://www.katacoda.com/loodse/courses/docker/docker-16-volumes)真是太棒了，循序渐进，又配有在线的 docker 环境，十分适合学习。

[另外一个网站](https://labs.play-with-docker.com)也有提供在线的 docker 环境，这个更牛逼，还可以直接访问云主机的端口，适合测试搭建 web 服务，一个 docker 容器能使用 4 个小时，过期了就重新申请，速度十分快，用于学习来说的话已经足够了，我主要就是通过这两个网站来学习 docker 知识的



## 运行一个容器



查看现有本地 docker 镜像

```bash
$ docker images
```



运行一个 docker 容器

```bash
$ docker run redis
```



将容器中的端口映射到主机的端口（前面是主机端口，后面是容器端口）

```bash
$ docker run -p 80:80 nginx
```



## 第一个容器



```bash
$ docker run ubuntu echo hello world
```

* run 运行一个新容器
* ubuntu 是 Linux 系统
* echo hello world 是要执行的命令



在容器中运行一个 apache 服务器

```bash
$ docker run -it -p 80:80 ubuntu
```

* 对于需要交互式的进程，用同时加上 `-i` 和 `-t`参数来为容器申请一个 tty ，通常直接写做 `-it`
* 要暴露容器中的端口的话，用 `-p` 或者 `-P` 参数来使得容器中的端口可以被宿主机以及任意一个可以访问宿主机的客户端访问（-P 是随机端口映射， -p 是指定端口映射）

上面命令输完之后，命令行就会切换成 ubuntu 容器的命令行，然后在 ubuntu 容器中安装 apache

```bash
$ apt-get update && apt-get install -y apache2
```

运行 apache 服务器

```bash
$ apache2ctl -DFOREGROUND
```

然后就可以在主机的 80 端口访问到 apache 服务器。此时，容器中的 apache 进程在前台执行，占用了一个命令行界面，要关闭的话直接按 `Ctrl+C` ，然后退出 ubuntu 容器按 `Ctrl+D` ，不过，这样的话，虽然容器停止了，但它在磁盘上还是存在着，可以用 `ps` 命令来查看，不带任何参数的 `ps` 只列出正在运行的容器，参数 `-a` 或者 `--all` 会将所有的容器都列出来

```bash
$ docker ps -a
```

要删除所有正在运行的和停止的容器可以用 `rm` 命令

```bash
$ docker rm $(docker ps -aq)
```



## 在 docker 中运行一个 Webapp



首先去找现有的镜像，直接用 `search` 命令可以在 DockerHub 上找到想要的镜像，可以直接搜索作者的名字

```bash
$ docker search loodse
```

![docker-search-name.jpg](https://i.loli.net/2019/12/16/6KwD9OuWTm2lrdE.jpg)

也可以直接搜索镜像的名字，比如 nginx 

```bash
$ docker search nginx
```

这样就会列出 DockerHub 上所有关于 nginx 的仓库，以及 star 数，注意镜像是以 `作者/程序` 的格式命名的，如果没有作者的话就说明这是官方的镜像

![docker-search-repo.jpg](https://i.loli.net/2019/12/16/g6PCEbSriGfeBc1.jpg)



找到之后我们就把镜像给拉取到本地，用 `pull` 命令

```bash
$ docker pull loodse/demo-www
```

![docker-pull.jpg](https://i.loli.net/2019/12/16/E6liypkTF8CmIj5.jpg)



这样本地就会多出一个镜像，我们可以用`dicker images` 命令查看，然后我们用 `-d` 选项在后台运行这个容器，`-d` 表示 detached 

```bash
$ docker run -d loodse/demo-www
```

输入上述命令之后容器运行就不会占用前台终端，只在后台运行，并且输出容器的 ID 号，用 `docker ps -a` 就可以看到这个容器正在运行

![container.jpg](https://i.loli.net/2019/12/16/9A3e8Qp6bdTjru4.jpg)



关于容器的一些信息：

| column       | meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| CONTAINER ID | 容器 ID，每个容器都都有一个唯一的 ID 作为标识符方便进行操作  |
| IMAGE        | 镜像的来源                                                   |
| COMMAND      | 执行的命令                                                   |
| CREATED      | 容器创建的时间                                               |
| STATUS       | 容器存在的状态                                               |
| PORTS        | 端口映射情况，这里没有做端口映射                             |
| NAMES        | 容器的名字，可以重命名为简单的名字方便记忆，同样，名字也是唯一的 |



我们可以用容器的 ID 号或者容器的名字来操作一个容器，例如，我们想要查看一个容器的详细信息，可以用 `inspect` 命令

```bash
$ docker inspect <container ID | container name>
```

![docker-inspect.jpg](https://i.loli.net/2019/12/16/x4odyqgRjpUFe9c.jpg)



我们如果想让正在运行的容器停止该怎么做呢，这里操作的是一个具体的容器，所以就要知道容器的 ID 号或者名字，直接用 `stop` 停止运行的容器，加上 `--time`  参数等待指定时间后停止容器

```bash
$ docker stop --time 5 <container ID | container name>
```

也可以用 `restart` 重启容器

```bash
$ docker restart <container ID | container name>
```

如果有容器被挂起了，也可以直接 `kill` 杀死容器进程

```bash
$ docker kill <container ID | container name>
```



## 与容器进行交互



让容器在后台运行可以用 `-d` 参数，如果要让一个在后台运行的容器转成前台可以用 `attach` 命令，比如我们用 `--name` 参数将容器命名为 counter1 并且在后台运行

```bash
$ docker run -d -it --name counter1 loodse/counter
```

这时在终端前台只会输出容器的名字，然后我们用 `attach` 命令将其转到前台运行，容器的标准输出就会附加在终端前台

```bash
$ docker attach counter1
```



## 交互式构建镜像



我们先从 DockerHub 上 pull 下来一个 debian 的镜像，并以交互式终端形式运行这个容器

```bash
$ docker run -it debian
```

然后我们在 debian 容器中安装 apache 服务器（很多情况下，在 docker 中 用 apt install 的话一定要加上 -y 选项）

```bash
$ apt-get update && apt-get install apache2 -y
```

之后我们退出这个容器，用 `docker ps -a` 命令来看看现有的容器，会找到刚刚退出的 debian 容器，记住它的 id 或者名字

![docker-diff-1.jpg](https://i.loli.net/2019/12/16/3Dyj8Q9zCZsmRoi.jpg)

我们接下来用 `diff` 命令看看他和之前的容器相比较有什么不同的

```bash
$ docker diff tender_wozniak
```

![docker-diff-2.jpg](https://i.loli.net/2019/12/16/cxVReZLgilq1Az6.jpg)

输出了一堆东西，这里截的不全，简单说下，前面的字母是有意义的，A 就代表 add，表示新增的文件，C 就代表改变的文件，D 就代表删除的文件。然后如果我们将这些改变提交的话就可以得到一个新的 docker 镜像，提交用的是 `commit` 命令，有没有发现，其实 docker 的操作和 git 是非常相像的！

```bash
$ docker commit tender_wozniak
```

![docker-commit-1.jpg](https://i.loli.net/2019/12/16/8GTcLmHxwOyfrng.jpg)

提交之后，终端就会输出新的镜像的 ID 号，此时输入 `docker images` 就可以发现多了一个镜像

```bash
$ docker images
```

![docker-commit-2.jpg](https://i.loli.net/2019/12/16/tF3Vybv2UBGHgEh.jpg)

我们来运行一个这个新的镜像，就进去了一个装好了 apache 的 debian 系统

```bash
$ docker run -it 9c0027df43f9
```

但是这种 ID 号很难记忆，我们可以用 `tag` 给新镜像打个标签，真的就跟 git 的操作是差不多的

```bash
docker tag 9c0027df43f9 webserver
```

打完标签就可以看到我们刚刚新创建的镜像变成了 webserver ，标签为 `latest`

![docker-tag.jpg](https://i.loli.net/2019/12/16/uUFs9aWzm8QEVP3.jpg)

我们就可以直接用这个名字来运行容器了

```bash
$ docker run -it webserver
```

![docker-tag-2.jpg](https://i.loli.net/2019/12/16/RZhxnleSCMY24Ud.jpg)



## 用 Dockerfile 来构建镜像



做 ctf  web 题的时候，就有很多出题人会在赛后将 web 题以 Dockerfile 的形式发布出来，复现题目，所以看懂 Dockerfile 是挺重要的，并且还得学会怎么写 Dockerfile



我们现在自己来用 Dockerfile 搭建一个镜像，先来看看我们要写入的 Dockerfile 里的内容

```dockerfile
FROM ubuntu:18.04
RUN apt-get update && apt-get install apache2 -y && apt-get clean
CMD ["apache2ctl", "-DFOREGROUND"]
```



来看看里面这些东西是啥意思

| command | meaning                                                      |
| ------- | ------------------------------------------------------------ |
| FROM    | 选择的镜像，默认是 latest 分支，想指定分支的话用 `:` 后加分支名称 |
| RUN     | 表示要执行的命令，不能有交互式的命令，因为在镜像构建的过程中无法用 stdin |
| CMD     | 表示执行的命令，有多个命令参数就用一个列表隔开来             |



然后我们就新建一个 Dockerfile 文件，注意 D 是大写的，把上面的东西写进去，有点像 Makefile 的感觉，反正就这样，我们在当前目录下输入下面这行命令

```bash
$ docker build -t webserver .
```

这行命令就是从 Dockerfile 创建镜像的命令，其中 `-t` 代表 tag，后面接的参数是镜像的名字或者标签，这里只给了名字，也可以叫做 `kevin/webserver:1.0` ，最后的 `.` 是代表从当前目录中寻找 Dockerfile 进行构建。



输入构建命令之后，docker 就按照我们的要求进行构建工作，他会按照我们 DOckerfile 里面的步骤来进行构建，会有 step1，step2，step3，因此最好将同一类的命令（例如 apt-get install）放在一行中写完，不然构建的效率会变低

![dockerfile.jpg](https://i.loli.net/2019/12/17/iRwQW8KEcsYLOVk.jpg)

构建完成后，我们就会多出一个镜像，这就是我们刚刚构建好的镜像

![dockerfile-2.jpg](https://i.loli.net/2019/12/17/DATQrlsK7ExFehN.jpg)

那我我们来运行一个这个镜像，他里面装的是 ubuntu 的 apache ，web 服务器默认是 80 端口，因此将他映射到主机的 8080 端口进行访问

```bash
$ docker run -p -d 8080:80 --name www webserver
```

可以看到，端口映射成功，我们能够访问到 apache 的默认页面

![docker-apache.jpg](https://i.loli.net/2019/12/17/pOFl3EWydnabrzh.jpg)

光有默认页面还不行，在 ctf 出题人做 docker 镜像的时候还会把自己写的文件给拷贝进去，接下来我们就来做这件事。我们在当前文件夹新建一个文件夹叫做 `html` ，在 `html` 文件夹中写入一个文件 `index.html` 

```html
<!DOCTYPE html>
<html>
  <head>
    <title>This is a title</title>
  </head>
  <body>
    <p>Hello world!</p>
  </body>
</html>
```

更新一下刚刚的 Dockerfile 文件

```dockerfile
FROM ubuntu:18.04
RUN apt-get update && apt-get install apache2 -y && apt-get clean
COPY html /var/www/html
CMD ["apache2ctl", "-DFOREGROUND"]
```

这里的 `COPY` 命令就是说将 `html` 文件夹中的内容拷贝到 docker 容器的 `/var/www/html` 里面，因为这是 apache 服务器的根目录。我们来重新构建一下这个镜像

```bash
$ docker build -t webserver .
```

运行容器并将其 apache 端口映射为主机的 8090 端口(8080 被刚刚的容器占用了)

```bash
$ docker run -d -p 8090:80 --name www1 webserver
```

然后浏览器打开主机的 8090 端口就可以访问到我们自定的 `index.html` ，或者也可以用 curl 这个牛逼的工具

```bash
[node1] (local) root@192.168.0.8 ~
$ curl localhost:4567
<!DOCTYPE html>
<html>
  <head>
      <title>This is a title</title>
  </head>
  <body>
  	<p>Hello world!</p>
  </body>
</html>
```



## 用 .dockerignore 忽略不要的文件



举个例子，当前目录下有一个 Dockerfile 文件，除了必要的文件外有一个机密文件和一个巨大的文件

```bash
$ ls 
cmd.sh	Dockerfile	password.txt	largefile.img
```

构建的过程中，我们的 Dockerfile 如果这样写的话会出现什么问题呢？（`ADD` 的作用和 `COPY` 类似，都是将宿主机目录下的文件传到 docker 容器对应的目录）

```dockerfile
FROM alpine
ADD . /app
COPY cmd.sh /cmd.sh
CMD ["sh", "-c", "/cmd.sh"]
```

没错，本地的敏感文件也会被传到镜像中，这样就可以引发安全问题，并且有些很大的我们完全不需要用到的文件也会被传进去，导致镜像的体积变得很大

```bash
$ docker build -t withpassword .
$ docker run withpassword ls /app
cmd.sh
Dockerfile
password.txt
largefile.img
```

为了解决这种情况，我们可以像 git 一样创建一个 `.dockerignore` 文件，将我们不想传递的文件给添加进去，这样就不会被传到镜像中

```bash
$ echo password.txtpassword.txt >> .dockerignore
$ echo largefile.img >> .dockerignore
```

这样子创建出来的镜像就不包含本地的敏感文件以及无关的文件

```bash
$ docker build -t no-passwd-no-large-file .
$ docker run no-passwd-no-large-file ls /app
Dockerfile
cmd.sh
```



## 管理 docker 日志



这部分没啥好讲的，就是 docker 容器在运行的过程中会产生日志文件，日志一多的话就会造成空间浪费，因此我们在运行容器的时候可以加上 `--log-driver` 参数选择日志驱动的类别



这是选择使用 syslog 来作为日志分类，syslog 会被系统集中管理，是个不错的选择

```bash
$ docker run -d --name redis-syslog --log-driver=syslog redis
```

当然，也可以选择不要日志输出，这样就没有日志文件

```bash
$ docker run -d --name redis-none --log-driver=none redis
```

可以使用 `logs` 命令来查看容器的日志

```bash
$ docker logs redis
```



不过 `--log-driver` ~~这个选项好像只有在类 Unix 系统才能使用，我在 Windows 上使用会报错~~，很多人都报错，网上没找到原因



## docker 网络基础



## 使用 volume 存储数据



在 `docker run` 后面加上参数 `-v` 可以指定一个主机的一个 volume 和 docker 容器的一个 volume ，使得两者可以共享文件。我们先来创造一下环境

![docker-volume.jpg](https://i.loli.net/2019/12/17/enjvVUw7XrKhZSE.jpg)

现在我要让 docker 能够访问宿主机上的 `/host-data` 里面的内容，输入以下内容

```bash
$ docker run -v /host-data:/data -it ubuntu
```

然后我们进入 docker 容器中，可以看到容器的 `/data` 文件夹中能够访问到宿主机的共享文件

![docker-volume-docker.jpg](https://i.loli.net/2019/12/17/hHOYWDFc7STjBXt.jpg)

不过默认情况下 docker 对宿主机的共享文件是由读写权限的，为了防止 docker 修改主机文件，我们可以对 docker 使用 `readonly` 选项

```bash
$ docker run -v /host-data:/data:ro ubuntu
```

这样的话就不能够对宿主机共享的文件进行写入操作了

![docker-readonly.jpg](https://i.loli.net/2019/12/17/xYOJ6hV4KsIZpvz.jpg)

