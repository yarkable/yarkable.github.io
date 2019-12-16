## preface



今早在做 web 题的时候，题目给的是 Dockerfile 文件，让我们自己搭建环境，然后刚好 Windows 上也装了 docker ，就打算在 Windows 上启动 docker 环境，然后忘了 Windows 上的 docker 需要启用 Hyper-V 环境，不然就没办法运行



![hyper-v.jpg](https://i.loli.net/2019/12/16/Yw8FXGObWqp23vi.jpg)



但是我 Windows 上又跑着 VMvare 虚拟机，这和 Hyper-V 是不能共存的，虽然我双系统 Ubuntu 上也有 docker，但是不想为了做个题目还切个系统，WSL 不香吗，然后就在 WSL 里面安装了 docker，讲道理，安装的过程爽得很，一步到位，但是输入命令的时候我就傻了 



```bash
$ docker run hello-world
```



这是用来测试的命令，判断 docker 是否安装成功，然而事情并没有这么简单，直接就给我摆了一道，运行不了



```bash
$ docker run hello-world
docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?. See 'docker run --help'.
```



这是 docker 的守护进程没有开启，后来上网查了一下，WSL 不支持 docker 的守护进程，我尼玛傻了都，解决方案就是用 WSL 中的 docker 连接 windows 的 docker-engine ，啥？？ 那我还是得开启 Hyper-V 啊，那这样还不如直接用 Windows ，果断卸载 ！



然后就一直不让我卸载，报错，原因就是 docker 这个服务已经停了，如果一个服务从来没有被运行过，则他无法被卸载，太狗了



```bash
sudo apt remove docker-ce
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following packages were automatically installed and are no longer required:
  containerd runc
Use 'sudo apt autoremove' to remove them.
The following packages will be REMOVED:
  docker-ce
0 upgraded, 0 newly installed, 1 to remove and 0 not upgraded.
After this operation, 62.7 MB disk space will be freed.
Do you want to continue? [Y/n] y
(Reading database ... 30768 files and directories currently installed.)
Removing docker-ce (1.13.1-0ubuntu1~16.04.2) ...
invoke-rc.d: could not determine current runlevel
 * Stopping Docker: docker
No process in pidfile '/var/run/docker-ssd.pid' found running; none killed.
invoke-rc.d: initscript docker, action "stop" failed.
dpkg: error processing package docker-ce (--remove):
 subprocess installed pre-removal script returned error exit status 1
dpkg: error while cleaning up:
 subprocess installed post-installation script returned error exit status 1
Errors were encountered while processing:
 docker-ce
E: Sub-process /usr/bin/dpkg returned an error code (1)
```



报错信息就是上面这样，网上搜了一下，有人给出了相似的经历，但是解决方案都不行，最终在 GitHub issue 上找到了一个较相似的解决方案，自己改了一下，就可以了



```bash
$ cd /var/lib/dpkg/info
$ sudo vim docker-ce.prerm
```



然后可以看到里面是这样的东西（我的已经被删除了，网上找的）



```bash
if ([ -x "/etc/init.d/docker" ] || [ -e "/etc/init/docker.conf" ]) && \
   [ "$1" = remove ]; then
        invoke-rc.d docker stop || exit $?
fi
```



直接将这一整个 if 语句给注释了（解决方案给出的是将 invoke 那一行注释，但是发现 if 语句会报错） ，再重新 `apt remove docker-ce` 就能够将 docker 给卸载了，还是老老实实开启 Hyper-V 或者用 Linux 吧



## reference



https://github.com/microsoft/WSL/issues/2702