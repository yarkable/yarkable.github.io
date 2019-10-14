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



## ssh-keyz



**Question** 



> As nice as it is to use our webshell, sometimes its helpful to connect directly to our machine. To do so, please add your own public key to ~/.ssh/authorized_keys, using the webshell. The flag is in the ssh banner which will be displayed when you login remotely with ssh to  with your username.



**Hint**



> (1) key generation tutorial [[1]](https://confluence.atlassian.com/bitbucketserver/creating-ssh-keys-776639788.html) 
>
>  (2) We also have an expert demonstrator to help you along. [link [2]](https://www.youtube.com/watch?v=3CN65ccfllU&list=PLJ_vkrXdcgH-lYlRV8O-kef2zWvoy79yP&index=4) 



**Solution**



看教程，在自己的 PC 上生成 ssh-key ，然后将公钥内容复制到服务器的 `./ssh/authorized_keys` 里面，以后就可以直接用 ssh 登录服务器不用输密码了，不添加的话可以连上服务器，但每次都要输入平台密码。



```shell
# 本地 PC 
$ ssh-keygen -t rsa -C "your_email@example.com"
$ ssh yarkable@2018picoctf.com
picoCTF{who_n33ds_p4ssw0rds_38dj21}  
```



```shell
# 服务器
$ mkdir .ssh
$ cd .ssh ; vim authorized_keys
```



## grep 1



**Question** 



> Can you find the flag in file [1] ? This would be really obnoxious to look through by hand, see if you can find a faster way. You can also find the file in /problems/grep-1_2_ee2b29d2f2b29c65db957609a3543418 on the shell server.



**Hint**



> Linux grep tutorial 



**Solution**



下载文件，或者 ssh 登录服务器，这里我用的是 ssh 登录的方式，直接进入到这个目录下面会发现有个 file 文件，**注意，不能先进入 problems 这个文件夹再进去 grep 那个文件夹，会说没有权限**，直接用 `cd` 命令进去，然后用管道命令在文本中搜索标准 flag 关键字即可



```shell
$ cd /problems/grep-1_2_ee2b29d2f2b29c65db957609a3543418                                 
$ cat file | grep pico                      
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
$ nc 2018shell.picoctf.com 49387                                                         
That wasn't so hard was it?                                                               
picoCTF{NEtcat_iS_a_NEcESSiTy_8b6a1fbc}    
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



## pipe



**Question** 



> During your adventure, you will likely encounter a situation where you need to process data that you receive over the network rather than through a file. Can you find a way to save the output from this program and search for the flag? Connect with 2018shell.picoctf.com 37542.



**Hint**



> (1) Remember the flag format is picoCTF{XXXX} 
>
> (2) Ever heard of a pipe? No not that kind of pipe... This [kind](http://www.linfo.org/pipes.html)



**Solution**



```shell
$ nc 2018shell.picoctf.com 37542 | grep pico
```



网速慢的话会等很久，我也就回寝室之前放着程序在那跑，然后第二天过来就拿到 flag 了。。



## grep 2



**Question** 



> This one is a little bit harder. Can you find the flag in /problems/grep-2_3_826f886f547acb8a9c3fccb030e8168d/files on the shell server? Remember, grep is your friend.



**Hint**



> Linux grep tutorial



**Solution**



这次进入目录之后会由很多文件夹，每一个文件夹里面又有很多个子文件夹，所以要用到 `grep` 的递归寻找，同时还可以直接用 `cat*/* ` 来遍历输出所有文件的内容再用 `grep` 命令搜索



```shell
$ grep -r pico                        files6/file16:picoCTF{grep_r_and_you_will_find_556620f7} 
$ cat */* | grep pico  # this is fine too
picoCTF{grep_r_and_you_will_find_556620f7} 
```



## Aca-Shell-A



**Question** 



> It's never a bad idea to brush up on those linux skills or even learn some new ones before you set off on this adventure! Connect with nc 2018shell.picoctf.com 27833.



**Hint**



> Linux for Beginners



**Solution**



这题不想吐槽。。vps 性能差的话根本就做不了题目，动不动就中断，重连了十几次才成功做出来。总之他会输出一些提示，按照他的提示输入基本的 Linux 命令就能够拿到 flag 



## environ



**Question** 



> Sometimes you have to configure environment variables before executing a program. Can you find the flag we've hidden in an environment variable on the shell server?



**Hint**



> [unix env](https://www.tutorialspoint.com/unix/unix-environment.htm) 



**Solution**



考察 Linux 环境变量，只需用用一条 `printenv` 命令就可以输出所有的环境变量



```shell
$ printenv | grep pico 
SECRET_FLAG=picoCTF{eNv1r0nM3nT_v4r14Bl3_fL4g_3758492}  
```



## you can't see me



**Question** 



> '...reading transmission... Y.O.U. .C.A.N.'.T. .S.E.E. .M.E.  ...transmission ended...' Maybe something lies in /problems/you-can-t-see-me_3_1a39ec6c80b3f3a18610074f68acfe69.



**Hint**



> (1) What command can see/read files? 
>
> (2) What's in the manual page of ls?



**Solution**



进去输入 `ls` ,发现是空的，可能有隐藏文件夹， `ls -al` 可以看到下面的东西，有个名叫 `.` 的隐藏文件，但是又不能用 `cat .` 来查看他里面的内容，因为 `.` 在 Linux 系统中有特殊含义



```shell
drwxr-xr-x   2 root       root        4096 Mar 25  2019 .                                 -rw-rw-r--   1 hacksports hacksports    57 Mar 25  2019 .                                 drwxr-x--x 556 root       root       53248 Mar 25  2019 ..  
```



可以用 `cat .*` 来查看



```shell
cat: .: Is a directory                                                                   cat: ..: Permission denied                                                               picoCTF{j0hn_c3na_paparapaaaaaaa_paparapaaaaaa_cf5156ef} 
```



## absolutely relative



**Question** 



> In a filesystem, everything is relative ¯\_(ツ)_/¯. Can you find a way to get a flag from this program [1] ? You can find it in /problems/absolutely-relative_0_d4f0f1c47f503378c4bb81981a80a9b6 on the shell server. Source [2] .



**Hint**



> (1) Do you have to run the program in the same directory? (⊙.☉)7
>
>  (2) Ever used a text editor? Check out the program 'nano'



**Solution**



这题考的是绝对目录和相对目录的区别，打开题目给的 C 程序分析



```c
#include <stdio.h>
#include <string.h>

#define yes_len 3
const char *yes = "yes";

int main()
{
    char flag[99];
    char permission[10];
    int i;
    FILE * file;


    file = fopen("/problems/absolutely-relative_0_d4f0f1c47f503378c4bb81981a80a9b6/flag.txt" , "r");
    if (file) {
    	while (fscanf(file, "%s", flag)!=EOF)
    	fclose(file);
    }   
	
    file = fopen( "./permission.txt" , "r");
    if (file) {
    	for (i = 0; i < 5; i++){
            fscanf(file, "%s", permission);
        }
        permission[5] = '\0';
        fclose(file);
    }
    
    if (!strncmp(permission, yes, yes_len)) {
        printf("You have the write permissions.\n%s\n", flag);
    } else {
        printf("You do not have sufficient permissions to view the flag.\n");
    }
    
    return 0;
}
```



可以观察到，flag 是以绝对目录的形式读入文件的， `permission.txt` 是相对路径，也就是在当前工作目录下的一个文件，通过分析程序逻辑，我们可以知道，当 permission 和 yes 前三个字符内容一样的时候就会输出 flag，利用这个我们可以在 home 目录下新建 `permission.txt` 文件，写入 ' yes ' ，然后直接在当前目录运行程序就拿到了 flag



```shell
$ pwd                                                                                     
/home/yarkable                                                                           
$ echo 'yes' > permission.txt                                                             
$ cat permission.txt                                                                     
yes                                                                                       
$ /problems/absolutely-relative_0_d4f0f1c47f503378c4bb81981a80a9b6/absolutely-relative   
You have the write permissions.                                                           
picoCTF{3v3r1ng_1$_r3l3t1v3_befc0ce1}       
```







