---
layout:     post
title:      picoCTF 2018 Writeup：General Skills
subtitle:   
date:       2019-10-14
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - writeup
---

## preface



在边学习 ctf 的过程中，大佬叫我们可以去做题目，边做题边积累，picoCTF 是面向美国高中生的 ctf 比赛，比较基础，适合我这种弱鸡，在做完了基础部分后觉得自己的认知增进不少，就写个 writeup 来记录一下解题过程，本篇文章应该会长期更新。



## General Warmup 1



**Question** 



> If I told you your grade was 0x41 in hexadecimal, what would it be in ASCII? 



**Hint**



> Submit your answer in our competition's flag format. For example, if you answer was 'hello', you would submit 'picoCTF{hello}' as the flag.



**Solution**



送分题，十六进制转 ASCII 码，在线工具或者 python 都可以



```python
chr(0x41)
'A'
```



## General Warmup 2



**Question** 



> Can you convert the number 27 (base 10) to binary (base 2)?



**Hint**



> Submit your answer in our competition's flag format. For example, if you answer was '11111', you would submit 'picoCTF{11111}' as the flag.



**Solution**



python 一行代码搞定。。 



```python
bin(27)
'0b11011'
```



## General Warmup 3



**Question** 



> What is 0x3D (base 16) in decimal (base 10).



**Hint**



> Submit your answer in our competition's flag format. For example, if you answer was '22', you would submit 'picoCTF{22}' as the flag.



**Solution**



依然是 python 。。

```python
0x3D
'61'
```



## Resources



**Question** 



> We put together a bunch of resources to help you out on our website! If you go over there, you might even find a flag! https://picoctf.com/resources



**Solution**



直接浏览器打开这个网站拉到最下面就是 flag



## grep 1



**Question** 



> Can you find the flag in file [1] ? This would be really obnoxious to look through by hand, see if you can find a faster way. You can also find the file in /problems/grep-1_2_ee2b29d2f2b29c65db957609a3543418 on the shell server.

**Hint**



> Linux grep tutorial 



**Solution**



下载文件，或者 ssh 登录服务器，这里我用的是 ssh 登录的方式，直接进入到这个目录下面会发现有个 file 文件，**注意，不能先进入 problems 这个文件夹再进去 grep 那个文件夹，会说没有权限**，直接用 `cd` 命令进去，然后用管道命令在文本中搜索标准 flag 关键字即可



```shell
$ cd /problems/grep-1_2_ee2b29d2f2b29c65db957609a3543418                                 $ cat file | grep pico                      
picoCTF{grep_and_you_will_find_42783683}     
```



## net cat



**Question** 



> Using netcat (nc) will be a necessity throughout your adventure. Can you connect to 2018shell.picoctf.com at port 49387 to get the flag?

**Hint**



> Linux nc tutorial 



**Solution**



考察的基本的 `nc` 命令，直接连上去就能拿到 flag



```shell
$ nc 2018shell.picoctf.com 49387                                                         That wasn't so hard was it?                                                               picoCTF{NEtcat_iS_a_NEcESSiTy_8b6a1fbc}    
```



## strings



**Question** 



> Can you find the flag in this file [1]  without actually running it? You can also find the file in /problems/strings_1_c7bac958dd6a4b695dc72446d8014f59 on the shell server.

**Hint**



> Linux strings tutorial 



**Solution**



考察 Linux 的 `strings` 命令，如果没有这个命令的要用 apt 包管理器安装一个 strings 包，`strings` 可以打印出文件中可打印的字符，然后送到管道里面用 `grep` 命令就拿到 flag 了



```shell
$ cd /problems/strings_1_c7bac958dd6a4b695dc72446d8014f59 
$ strings strings | grep pico              
picoCTF{sTrIngS_sAVeS_Time_d7c8de6c}       
```



