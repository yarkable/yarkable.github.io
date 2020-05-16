---
layout: post
title: Docker安装OpenCV
subtitle: 
date: 2020-05-15
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - docker
    - OpenCV
---



## preface

以前装OpenCV都是在自己物理机编译安装，最近部署深度学习模型需要用到OpenCV，但是模型要部署在服务器上，而服务器有很多人一起使用，不能安装在系统环境中(主要我也没有权限安装)，有种方案就是在 docker 里面装一个 OpenCV 进行调用。讲道理，我只在 docker 里面装过 web 应用，OpenCV 这种不需要端口映射的还是第一次装，所以就记一下吧。



## 拉 docker 镜像

之前我还以为是用 docker pull 一个 ubuntu 镜像然后在里面装 OpenCV ，师兄跟我说有现成的 OpenCV docker 镜像直接用就行了，直接在命令行中输入下面的命令就会搜索到很多关于 OpenCV 的镜像，但是 `docker search` 这个命令无法获取到镜像的详细信息，默认是 pull 最终版本，如果我们想自己指定下载 Tag 版本号的镜像就要上[官网](https://hub.docker.com/)看看具体的版本信息

```bash
$ docker search opencv
```



我下载的是 OpenCV3.4.3 版本的，直接按照提示的命令下载就行了，20 多 G，下载有点久…



![docker-hub](https://i.loli.net/2020/05/15/YRrGOBN6PjufEb1.png)



然后发现它报错了，报错信息如下：

```bash
failed to register layer: Error processing tar file(exit status 1): write /opt/opencv/build/modules/flann/test_precomp.hpp.gch/opencv_test_flann_RELEASE.gch: no space left on device
```



乍得一看还以为服务器没磁盘空间了，要不是服务器有几个 TB 的容量我差点就信了，应该是 docker 的问题，上网查了一下，说是因为  docker 没空间了，删掉一些镜像就行了，在GitHub找到了[解决方案](https://github.com/moby/moby/issues/32811)，用 `docker system df` 命令查看镜像占用的容量，再用 `docker system prune -a` 命令删除所有没有利用到的空间(这个命令把我所有的 docker 镜像全删了，最好还是不要乱用，想办法给 docker 扩容更好)



重新下载，完事之后按照下面官方给的命令打开容器进行操作

```bash
$ docker run --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8888:8888 -it spmallick/opencv-docker:opencv /bin/bash
```



但是也报错了，原因是服务器的 8888 端口已经被我的 jupyter 服务给占用了，因此要换一个端口进行映射，并且服务器上也没有 video0 这个设备，所以把 `--device` 选项删除就可以了，换成下面的命令就可以进去 docker 容器了

```bash
$ docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8889:8888 -it spmallick/opencv-docker:opencv-3.4.3 /bin/bash
```



其实这个容器就是一个 ubuntu 系统，上面装好了 OpenCV 以及其他的编译工具，在 `/usr/local/lib` 里面找到了 OpenCV 的链接库



![libs](https://i.loli.net/2020/05/16/GegrIwH3FWaNEqA.png)



But，默认的用户是 jovyan，这个用户是没有开启密码的，也就是说没有 sudo 权限，不能够下载东西，就很蛋疼，[上网查了一下](https://github.com/kubeflow/kubeflow/issues/425)，这样做是为了避免用户从公共主机上的映像启动容器的情况。但是，有一个叫 `GRANT_SUDO`的环境变量可以传递给容器来完成 sudo 任务，也就是在启动容器的时候传递个环境变量给容器

```bash
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e GRANT_SUDO=yes --user root -p 5000:5000 -p 8889:8888 -it spmallick/opencv-docker:opencv-3.4.3 /bin/bash
```



cmake -DCMAKE_PREFIX_PATH=/libtorch ..



docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8889:8888 -e GRANT_SUDO=yes --user root -v ~/kevin/code/pytorch/pytorch-model-deployment:/pytorch-deployment -v ~/kevin/code/pytorch/libtorch:/libtorch -it spmallick/opencv-docker:opencv-3.4.3 /bin/bash