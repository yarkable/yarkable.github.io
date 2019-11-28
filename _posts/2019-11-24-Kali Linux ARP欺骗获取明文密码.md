---
layout:     post
title:      Kali Linux ARP欺骗获取明文密码
subtitle:   http
date:       2019-11-24
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - security
---



## preface



想起来之前总是听别人说公共场合的 WiFi 不要乱连，会泄露隐私信息，一直不知道怎么会泄露，最近学了点网络安全的知识就自己做了个实验，来看看是否真的能抓到数据。



**实验平台：**

靶机：windows 10 物理机

攻击机：Kali Linux 虚拟机



整个网络拓扑如下：

![tuopu.jpg](https://i.loli.net/2019/11/28/tQxdVec5sBfuT34.jpg)

**本篇文章纯粹为了提高人们的安全意识，切勿用作非法用途**



## ARP 协议



先来简要的说一下啊 ARP 协议是什么，简单来说就是地址解析协议(**A**ddress **R**esolution **P**rotocol)，在网络通信中，两台机器之间要进行通信的话必须要知道对方的 MAC 地址，在网络层一般只关心通信主机的 ip地址，这就导致在以太网中使用 IP 协议时，数据链路层的以太网协议接到上层 IP 协议提供的数据中，只包含目的主机的 IP 地址，这就需要 ARP 协议来将 IP 地址解析为对应的 MAC 地址。



>  ARP 欺骗就是利用了这一协议，其运作原理是由攻击者发送假的ARP数据包到网络上，尤其是送到网关上。其目的是要让送至特定的 IP 地址的流量被错误送到攻击者所取代的地方。因此攻击者可将这些流量另行转送到真正的网关（被动式数据包嗅探，passive sniffing）或是篡改后再转送（中间人攻击，man-in-the-middle attack）。攻击者亦可将ARP数据包导到不存在的[MAC地址](https://zh.wikipedia.org/wiki/MAC位址)以达到拒绝服务攻击的效果，例如[netcut](https://zh.wikipedia.org/w/index.php?title=Netcut&action=edit&redlink=1)软件



## ettercap



这是 Kali Linux 下自带的一款 ARP 欺骗工具，这里我们就讲下基本使用，基本原理以后再专门写一篇，它能够扫描出局域网中活跃的主机并且进行 ARP 欺骗，首先确保我们攻击机有一块网卡已经连上了局域网，之后就按步骤进行，首先在 Kali 命令行打开 ettercap 图形界面



![ettercap.jpg](https://i.loli.net/2019/11/27/FbQz8mqJBPnhcjY.jpg)



然后开启 `unified sniffing` 



![sniffing.jpg](https://i.loli.net/2019/11/27/2XGy4Dm9LH7zMcV.jpg)



选择一个网络设备进行欺骗，因为我是用网卡 WiFi 连接我的路由器，所以我这里选择的是 wlan0



![networks.jpg](https://i.loli.net/2019/11/27/EcIDrMYC4ZBhgXz.jpg)



紧接着扫描局域网中的网络设备，扫描完的设备可以打开 `Hosts list` 查看



![scan.jpg](https://i.loli.net/2019/11/27/gtzqaiMXZnd53sR.jpg)



![hosts.jpg](https://i.loli.net/2019/11/27/F45t2PJuGgbCUIN.jpg)



可以看到我的路由器，物理机和 Kali Linux 攻击机的局域网 ip 地址，我们将攻击的目标放到 `Target1` ，将路由器/网关放到 `Target2`，准备开始嗅探



![target.jpg](https://i.loli.net/2019/11/27/7GlS9LRZJ5thwzE.jpg)



点击 `Mitm` 面板的 `ARP poisoning` 



![arp-poison.jpg](https://i.loli.net/2019/11/27/teZdBLjPuJbNUFn.jpg)



勾选住 `sniff remote connections` 点击 OK 即可



![remote-sniffer.jpg](https://i.loli.net/2019/11/27/KSdjZtE3bflxmiX.jpg)



开始嗅探，然后靶机的流量就会首先经过我们的攻击机了



![start-snifing.jpg](https://i.loli.net/2019/11/27/9yafSrLFYMjEBWw.jpg)



## wireshark抓包



开启了嗅探之后，靶机上的所有流量都会先流经我们的攻击机再传输出去，相当于我们是个中间人，因此，我们就可以用 wireshark 抓靶机上流量包来获取一些敏感信息，我在物理机上面登录我们学校的在线 OJ 平台，这是用 HTTP 加密的，因此所有信息都是明文传输，我们可以获取到账号和密码



![ojj.jpg](https://i.loli.net/2019/11/28/SHOBdavDAKYIp49.jpg)



我们在 wireshark 里面抓取经过 wlan0 的流量包，也就是靶机上发过来的流量包，过滤出 HTTP 协议流量，提交表单一般用的是 POST 方法，因此直接过滤出 HTTP 中的 POST 请求，可以看到，账号密码一览无余。其他协议也是一样的，反正靶机上的所有流量都会经过我们的攻击机，只是有些协议像 HTTPS 之类的会进行加密，所以很难破解



![passwd.jpg](https://i.loli.net/2019/11/28/zTIEysivSeHun8o.jpg)



## 后话



校园网是个巨大的局域网，为我们带来了很多便携，但是同时也造成了很多安全隐患，例如我们的校园 WIFI 也是没有加密的，如果连上去的话很可能就会被别人嗅探到隐私信息，因此一般提示有 `不安全` 字样的 WIFI 最好不要连，在外面的不明 WIFI 也千万不要连，要时刻注意互联网安全





## reference 



[https://zh.wikipedia.org/wiki/ARP%E6%AC%BA%E9%A8%99](https://zh.wikipedia.org/wiki/ARP欺騙)

https://www.idaobin.com/archives/999.html

