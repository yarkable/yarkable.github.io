---
layout:     post
title:      Linux crontab创建定时任务
subtitle:   
date:       2019-12-18
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
---





## preface 



最近在学校论坛上看到一个脚本，用路由器固件自动拨号，解决断网的问题，其中就用到了 `crontab` 这个命令，大致原理就是用 `curl` 去抓取上网登录页面，在线和不在线的页面是不同的，根据这个特征用 `grep` 命令正则筛选出来，每分钟运行一次脚本，如果掉线的话，就将登陆账号和密码通过 POST 请求发送给页面，登陆成功就能上网



```bash
* * * * * username="校园卡号" && password="统一身份认证密码" && if [ $(curl -k https://drcom.szu.edu.cn | grep -oE WebLoginID_1) != "WebLoginID_1" ]; then curl -k https://drcom.szu.edu.cn/a70.htm --data "DDDDD=$username&upass=$password&0MKKey=%B5%C7%A1%A1%C2%BC"; fi
```



## at



在生活中，有很多事情都是周期性进行的，每天要吃饭睡觉洗澡，也有突发性的任务，上午接到老板通知下午开会，就得定个日程规划防止忘记，类比 Linux 系统也是一样的，有很多都是定期进行的任务，比如 ubuntu 隔几天就会提示我们是否升级等等，这是通过什么机制的呢？就是 `at` 和 `crontab` 这两个东西咯



`at` 命令呢运行的就是一次性的事件，如果说想让计算机提醒我们一个小时后记得喝水，用的就是 `at` 命令，不过，在 ubuntu 中，这个命令默认是没有开启的，它是由 `atd` 这个服务来提供的，所以我们要去激活 `atd` 才能使用这项功能

```bash
$ /etc/init.d/atd start
```



使用方法就是很粗暴，加上时间参数和任务事项就行了(五分钟后执行一个 python 脚本)

```bash
kevin@laptop:~$ at now + 5 minutes
warning: commands will be executed using /bin/sh       
at> /usr/bin/python3 demo.py
at> <EOT>
job 1 at Wed Dec 18 15:39:00 2019 
```



不过我这里 WSL 运行不了 atd ，所以就不演示了，毕竟今天主要讲的是 crontab ，定时的任务才能自动化嘛



## crontab



>  相对於 at 是仅运行一次的工作，循环运行的例行性工作排程则是由 cron (crond) 这个系统服务来控制的。刚刚谈过 Linux 系统上面原本就有非常多的例行性工作，因此这个系统服务是默认启动的。另外，由於使用者自己也可以进行例行性工作排程，所以咯， Linux 也提供使用者控制例行性工作排程的命令 (crontab)



### 安全性



想要创建周期性的任务事项时，使用 crontab 这个命令，但是得了安全起见，我们可以限制能够使用 crontab 的使用者账号（避免服务器被黑了，然后黑客随意创建定时任务定时盗取数据）



我们可以用两个方法做到这一点

1. `/etc/cron.allow `

   将可以使用 crontab 的账号写进去，不在这个文件内的使用者不能使用 crontab

2. `/etc/cron.deny`

   将不可以使用 crontab 的账号写进去，在这个文件内的使用者不能使用 crontab

实际上，上面讲的 at 也有这个功能。以优先顺序来说， `/etc/cron.allow ` 要比 `/etc/cron.deny` 优先级更高，而在我们自己判断时，只需要用一个文件即可，系统默认保留的是 `/etc/cron.deny` （虽然我的 ubuntu 系统并没有）



### 如何配置



以下是创建 crontab 周期任务的命令

```bash
[root@www ~]$ crontab [-u username] [-l|-e|-r]
-u  ：只有 root 才能进行这个任务，亦即帮其他使用者创建/移除 crontab 工作排程；
-e  ：编辑 crontab 的工作内容
-l  ：查阅 crontab 的工作内容
-r  ：移除所有的 crontab 的工作内容，若仅要移除一项，请用 -e 去编辑。
```



只要我们不是在 `/etc/cron.deny` 名单中，我们就可以输入上面的命令来创建任务。范例一：用 dmtsai 的身份在每天的 12:00 发信给自己

```bash
[dmtsai@www ~]$ crontab -e
# 此时会进入 vi 的编辑画面让您编辑工作！注意到，每项工作都是一行。
0   12  *  *  * mail dmtsai -s "at 12:00" < /home/dmtsai/.bashrc
#分 时 日 月 周 |<==============命令串========================>|
```



看到前面有很多的 `*` ，下面我们解释一下这些东西是什么意思



| 代表意义 | 分钟 | 小时 | 日期 | 月份 | 周   | 命令       |
| -------- | ---- | ---- | ---- | ---- | ---- | ---------- |
| 数字范围 | 0-59 | 0-23 | 1-31 | 1-12 | 0-7  | 呀就命令啊 |



因此 crontab 命令前面有五个时间选项确定命令在什么时候被进行，有意思的是， 在 **周** 这个字段中， 0 或 7 都代表着星期天！除此之外，还有一些辅助的字符来更细的划分时间

| 特殊字符 | 代表意义                                                     |
| :------- | :----------------------------------------------------------- |
| *        | 代表任何时刻都接受的意思！举例来说，范例一内那个日、月、周都是 * ，就代表著 **不论何月、何日的礼拜几的 12:00 都运行后续命令 ** 的意思！ |
| ,        | 代表分隔时段的意思。举例来说，如果要下达的工作是 3:00 与 6:00 时，就会是：0 3,6 * * * command 时间参数还是有五栏，不过第二栏是 3,6 ，代表 3 与 6 都适用！（不能加空格） |
| -        | 代表一段时间范围内，举例来说， 8 点到 12 点之间的每小时的 20 分都进行一项工作：20 8-12 * * * command 仔细看到第二栏变成 8-12 喔！代表 8,9,10,11,12 都适用的意思！ |
| /n       | 那个 n 代表数字，亦即是『每隔 n 单位间隔』的意思，例如每五分钟进行一次，则：*/5 * * * * command 很简单吧！用 * 与 /5 来搭配，也可以写成 0-59/5 ，相同意思！ |



理解了上面这些的话就可以自己实现一个定时任务了，这里还是再用个例子来加深理解。假若你的女朋友生日是 5 月 2 日，你想要在 5 月 1 日的 23:59 发一封信给他，这封信的内容已经写在 /home/dmtsai/lover.txt 内了，该如何进行？

```bash
$ crontab -e
59 23 1 5 * mail kiki < /home/dmtsai/lover.txt
```



后面的命令最好用绝对路径，防止有错，建立好定时任务后我们就可以用 `crontab -l` 来查看当前有哪些定时任务在进行了，这是我一个 docker 容器中的定时任务（这里已经将参数给分开来方便查看意思）



```bash
# do daily/weekly/monthly maintenance
# min   hour    day     month   weekday command
*/15    *       *       *       *       run-parts /etc/periodic/15min
0       *       *       *       *       run-parts /etc/periodic/hourly
0       2       *       *       *       run-parts /etc/periodic/daily
0       3       *       *       6       run-parts /etc/periodic/weekly
0       5       1       *       *       run-parts /etc/periodic/monthly
```



## reference

[《鸟哥的 Linux 私房菜》](http://cn.linux.vbird.org/linux_basic/0430cron.php#crontab)