---
layout:     post
title:      Hyper + oh_my_zsh 配置最炫终端
subtitle:  	帅到爆炸，颜值即正义啊！
date:       2019-10-16
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux

---



## preface



之前在 Ubuntu 底下做开发时一直用的默认终端，其实感觉默认的终端还是有点难看，目前日常就是在 Windows 上用 WSL 的 Linux 环境，不得不说，微软真牛逼！WSL 太香了，搞得我都不想再去用回双系统了，该有的 WSL 里面基本都有了。好了，言归正传，要想让我舒服的使用 WSL ，那肯定得先配置得漂漂亮亮的，毕竟颜值才是第一生产力！



## 安装Hyper



[Hyper](https://hyper.is/) 是我最新发现的一款终端模拟器，是用 Electron 开发的，所以在 WIN/Mac/Linux 上都能够使用，我的测试环境为 Windows 10，基于 Electron 开发的产品都有个毛病，那就是很吃内存，不过我觉得还是可以接受的，毕竟有颜值就是任性。



下面是官网的截图，看上去就炫酷十足，不说了，赶紧整一个吧



![](C:\Users\kevin\Desktop\blog\hyper-official.jpg)



### 更换默认 shell



下载好了后直接打开就是 Windows 的 cmd 界面，但这玩意真的是不好用，功能很少，所以我们就把默认的 shell 换成 Windows 的 powershell 。打开 `edit->preference` 就是 Hyper 的配置选项，是一个名叫 `.hyper.js` 的 js 脚本，把 shell 那里按照上面的提示改成 powershell ，然后 `view->reload` ， 重新打开 Hyper 默认就是 powershell 了。



![](C:\Users\kevin\Desktop\blog\cfg.jpg)



基本所有的配置都是在这个文件里面更改的，包括字体，终端，插件管理，样式等等



![](C:\Users\kevin\Desktop\blog\cfg.gif)



### 下载Hyper插件



Hyper 的生态圈特别活跃，拥有特别多的插件，官网也推荐了一些下载量很高的插件，还有很多非常漂亮的主题，基本每个人都能找到自己喜欢的主题



![](C:\Users\kevin\Desktop\blog\plugin.gif)



我下载了 `hyperpower` 这个插件，这个最早是在 Atom 编辑器中被开发出来的，我一直都有在用，包括 VSCode 和 Pycharm 都有这种插件，效果十分炫酷，默认为 `power` 模式，爆炸块颜色单一并且窗口不晃动，输入 `wow` 之后进入 `wow` 模式，变成彩色色块且窗口晃动，再次输入则变回 `power` 模式。不知道是不是 bug，当我输入 `clear` 或者从 `vim` 中退出来的时候，他会自动变成 `power` 模式，让我很不爽。。



![](C:\Users\kevin\Desktop\blog\theme.jpg)



## 下载 zsh



我们拿 Hyper 是要用他来打开 WSL 的，WSL 的默认 shell 是 Bash， 虽然也挺好用的，但是 zsh 更加不错，整一个



```shell
$ sudo apt-get install zsh
```



安装完成后会提示是否要将 zsh 变成默认的 shell ，选择是即可，如果不想现在就更改，后续也可以输入以下命令进行 shell 的更改



```shell
chsh -s /bin/zsh
```



### 下载 oh-my-zsh



光有 zsh 还不够，因为默认的 zsh 并说不上炫酷，只是功能多了一点而已，要想终端变得炫酷还要装上 oh-my-zsh 对 zsh 进行快速配置，只用下面的命令就能下载，然后就发现 zsh 变了个主题。



```shell
$ sh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
```



觉得默认的主题不好看的话没关系，oh-my-zsh 自带了十分多的主题，具体主题的名称和截图可以去[这里](https://github.com/robbyrussell/oh-my-zsh/wiki/Themes)查看，修改主题要编辑主目录下的  `.zshrc` ，找到 `ZSH_THEME=` 这一行，然后修改为你喜欢的主题（我的主题是 ys，效果如下）



```shell
ZSH_THEME="ys"
```



![](C:\Users\kevin\Desktop\blog\ys_theme.jpg)



### 下载 zsh 插件



zsh 社区还有人开发了一堆很好用的插件，这里就介绍几个







