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



我们现在自己来用 Dockerfile 搭建一个镜像，先来看看我们的 Dockerfile 里的内容

```dockerfile
FROM ubuntu:18.04
RUN apt-get update && apt-get install apache2 -y && apt-get clean
CMD ["apache2ctl", "-DFOREGROUND"]
```

