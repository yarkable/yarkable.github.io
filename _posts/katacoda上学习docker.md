## 运行一个容器



查看现有本地 docker 镜像

```bash
$ docker images
```



运行一个 docker 容器

```bash
$ docker run redis
```



将容器中的端口映射到主机的端口

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
* 要暴露容器中的端口的话，用 `-p` 或者 `-P` 参数来使得容器中的端口可以被宿主机以及任意一个可以访问宿主机的客户端访问

上面命令输完之后，命令行就会切换成 ubuntu 容器的命令行，然后在 ubuntu 容器中安装 apache

```bash
$ apt-get update && apt-get install -y apache2
```

运行 apache 服务器

```bash
$ apache2ctl -DFOREGROUND
```

然后就可以在主机的 80 端口访问到 apache 服务器。此时，容器中的 apache 进程在前台执行，占用了一个命令行界面，要关闭的话直接按 `Ctrl+C` ，然后退出 ubuntu 容器按 `Ctrl+D` ，不过，这样的话，虽然容器停止了，但它在磁盘上还是存在着，可以用 `ps` 命令来查看

```bash
$ docker ps -a
```

要删除所有正在运行的和停止的容器可以用 `rm` 命令

```bash
$ docker rm $(docker ps -aq)
```



## 在 docker 中运行一个 Webapp





