---
layout: post
title: Jekyll 渲染 LaTeX 数学公式
subtitle: 
date: 2019-12-24
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - LaTeX
    - troubleshoot
---



## preface



最近又在看深度学习相关的内容，并且讲学习笔记呈现在了博客上，课程中有很多的公式，之前都是靠着截图的，但是这样就太繁琐了，想着如果能够直接敲公式的话效率就会高点了，就去看了一下 LaTeX 的语法，好像挺简单的，就稍微学习了一下，想发布到博客上发现渲染不出来



## solution



博客是用 MarkDown 编写的，而 MarkDown 本身并不支持公式，但是我使用的 Typora 编辑器可以扩展 LaTeX 的公式，只要勾选 `文件 -> 偏好设置 -> MarkDown -> 内联公式` ，然后再重启 Typora 就可以编辑行内的 LaTeX 公式了



![formula.jpg](https://i.loli.net/2019/12/25/7TMFeKk2rI5iH4W.jpg)



下面是在 Typora 上编辑公式展示的效果



![inline.jpg](https://i.loli.net/2019/12/25/ekZNVEwmrsh3Uly.jpg)

![block.jpg](https://i.loli.net/2019/12/25/Dpkerw8ma2sCvJ5.jpg)



这在 Typora 里面是没有任何问题的，可以正常显示，但是 push 到基于 Jekyll 模板的 GitHub page 上就出现了不能渲染公式的问题，网上搜了一下，Jekyll 模板确实不支持 LaTeX 公式，不过有一种解决方法，那就是引入外部的 js脚本，可能会影响一点加载速度，但是我也没有什么明显的感觉(可能因为我挂着全局代理)



具体方法就是，到 `_config.yml` 中加上一行 `markdown: kramdown` ，再打开 `_includes` 文件夹中的 `head.html` ，将下面这段脚本粘贴进去，提交修改，就可以正常渲染 LaTeX 公式了



```html
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
        }
    });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```



下面分别是我的内联公式和公式块的测试效果



```latex
$y = \sin(\pi + \theta)$
```



$ y = \sin(\pi + \theta)$



```latex
$$
\begin{align*}
y = y(x,t) &= A e^{i\theta} \\
&= A (\cos \theta + i \sin \theta) \\
&= A (\cos(kx - \omega t) + i \sin(kx - \omega t)) \\
&= A\cos \frac{2\pi}{\lambda} (x - v t) + i A\sin \frac{2\pi}{\lambda} (x - v t)
\end{align*}
$$
```


$$
\begin{align*}
y = y(x,t) &= A e^{i\theta} \\
&= A (\cos \theta + i \sin \theta) \\
&= A (\cos(kx - \omega t) + i \sin(kx - \omega t)) \\
&= A\cos \frac{2\pi}{\lambda} (x - v t) + i A\sin \frac{2\pi}{\lambda} (x - v t)
\end{align*}
$$


## reference



https://stackoverflow.com/questions/26275645/how-to-support-latex-in-github-pages





