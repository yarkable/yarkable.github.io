---
layout:     post
title:      Macbook配合VSCode和MacTeX本地编译LaTeX
subtitle:   
date:       2022-09-30
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - troubleshoot

---



### preface



基于 overleaf 在线写 latex 太麻烦了，每次都需要重新编译一下要等很久，本地的话就很快。之前在 windows 上有试过 vscode 插件加上 texlive 进行本地编译，现在主要用的是 MacBook，所以记录一下在 MacBook 上通过 vscode 插件加上 latex 编译器实现本地的编译。



### 需要的东西



#### vscode 插件： latex workshop

直接插件市场搜索安装就行

#### latex 编译器

大家选的都是 mactex，有两种安装方式，一种是[官网下载 pkg](https://media.icml.cc/Conferences/CVPR2023/cvpr2023-author_kit-v1_1-1.zip)，一种是用 brew 安装。

```bash
brew install --cask mactex-no-gui
```

装好之后将可执行程序添加到 $PATH 当中，不然会找不到

```bash
vim ~/.bash_profile
export PATH=/Library/Tex/texbin:$PATH
source ~/.bash_profile
```

然后这里就完事了，接下去去配置 vscode



### VSCode 配置



在 setting 的 json 里面输入下面内容(command + shift + p)

```json
"latex-workshop.latex.tools": [
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "%DOC%"
            ]
        },
        {
            "name": "cd",
            "command": "cd",
            "args" : ["%DIR%"]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DOCFILE%"
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "latexmk",
            "tools": [
                "latexmk"
            ]
        },
        {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
                "cd",
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        }
    ],
```

然后就完事了，command + option + b 编译，command + option + v 查看 pdf



### 遇到的坑



因为我是将整个项目都放在 iCloud 里面方便进行同步的，但是这样会报错

```txt
Latexmk: Filename '/Users/bytedance/Library/Mobile Documents/com~apple~CloudDocs/my_work/CrossDataset/cvpr2023-author_kit-v1_1-1/latex/PaperForReview' contains character not allowed for TeX file.
Latexmk: Stopping because of bad filename(s).
Rc files read:
  NONE
Latexmk: This is Latexmk, John Collins, 17 Mar. 2022. Version 4.77, version: 4.77.

```

查看了 GitHub 上的 issue 发现是因为 Apple 对 iCloud 文件夹会添加一些奇怪的字符，一种曲线救国的方法就是给 iCloud 生成一个软链接，然后从软链接进去就能解决这个问题（一定要从软链接的根目录进去，不能从子文件夹进去，否则还是会报错）

```bash
ln -s /Users/xxx/Library/Mobile\ Documents/com~apple~CloudDocs/ iCloud
code iCloud/my_work/xxx
```



### reference



https://github.com/James-Yu/LaTeX-Workshop/issues/234

https://blog.csdn.net/qq_31460257/article/details/81592812

https://zhuanlan.zhihu.com/p/102823687