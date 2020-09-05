---
layout: post
title: 配置并美化Windows terminal
subtitle: 
date: 2020-09-05
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - shell
---



## preface



今天配置一下据说很强大的 Windows terminal，因为刚刚下载的时候是挺丑的，还是黑黑的框，得自己去折腾一下。



## 配置 powershell



Windows terminal 默认的终端是 powershell，所以有必要把 powershell 给配置一下，仿照 oh-my-zsh 的主题别人开发出了 oh-my-posh，直接用几个命令把包给下载下来就行了。



```powershell
Install-Module posh-git -Scope CurrentUser
Install-Module oh-my-posh -Scope CurrentUser
```



如果以上命令报错则用管理员权限打开 powershell，输入以下命令就可以了



```powershell
Set-ExecutionPolicy RemoteSigned
```



装好之后运行下面命令打开一个 powershell 脚本文件



```powershell
if (!(Test-Path -Path $PROFILE )) { New-Item -Type File -Path $PROFILE -Force }
notepad $PROFILE
```



输入下列命令，以后 powershell 启动的时候就会运行这些命令加载模块



```powershell
Import-Module posh-git
Import-Module oh-my-posh
Set-Theme  Pararussel
```



OK，这就算初步配置完成了，此时肯定是乱码的，因为这种主题需要 Powerline 字体，这里推荐 JetBrains Mono 字体，不过暂时还不用改，后面会统一修改字体



## 配置 terminal



然后就到了配置 Windows terminal 了，其实也挺简单的，terminal 的配置文件就是一个 json 文件，使用 vscode 打开就行了。系统默认为 cmd 和 powershell 以及 wsl 已经配置好了，自己可以修改，如果想要加其他的选项就可以照葫芦画瓢添加，这里贴一张少数派文章里的图，展示了 json 的不同功能



![](https://cdn.sspai.com/editor/u_spencerwoo/15836861908691.png?imageView2/2/w/1120/q/90/interlace/1/ignore-error/1)



只要在 profile 里面新增自己的 profile 就行了，但是要注意不能用同样的 GUID，可以在 powershell 中用 ` new-guid` 命令生成一个独特的 GUID ，然后配色主题的话可以去 [GitHub 这个网站](https://github.com/mbadolato/iTerm2-Color-Schemes)找到，然后贴到 `schemes` 里面就行了。



有不清楚的配置直接去 Windows terminal 官网或者少数派的博客上看一下，这里贴上我自己的配置，方便下次直接抄就行了



```txt
{
    "$schema": "https://aka.ms/terminal-profiles-schema",

    "defaultProfile": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",

    // You can add more global application settings here.
    // To learn more about global settings, visit https://aka.ms/terminal-global-settings

    // If enabled, selections are automatically copied to your clipboard.
    "copyOnSelect": true,

    // If enabled, formatted data is also copied to your clipboard
    "copyFormatting": false,

    "theme": "dark",
    // A profile specifies a command to execute paired with information about how it should look and feel.
    // Each one of them will appear in the 'New Tab' dropdown,
    //   and can be invoked from the commandline with `wt.exe -p xxx`
    // To learn more about profiles, visit https://aka.ms/terminal-profile-settings
    "profiles":
    {
        "defaults":
        {
            // Put settings here that you want to apply to all profiles.
            "fontFace": "JetBrains Mono",
        },
        "list":
        [
            {
                // Make changes here to the powershell.exe profile.
                "background": "#013456",
                "acrylicOpacity": 0.8,
                "useAcrylic": true,
                "fontSize": 10,
                "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
                "name": "Windows PowerShell",
                "commandline": "powershell.exe",
                "hidden": false
            },
            {
                // Make changes here to the cmd.exe profile.
                "guid": "{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
                "name": "cmd",
                "commandline": "cmd.exe",
                "hidden": false
            },
            {
                "acrylicOpacity": 0.8,
                "useAcrylic": true,
                "colorScheme": "Ubuntu",
                "icon": "D:/assets/img/png/icons8-ubuntu.png",
                "guid": "{2c4de342-38b7-51cf-b940-2309a097f518}",
                "hidden": false,
                "name": "Ubuntu",
                "source": "Windows.Terminal.Wsl"
            },
            {
                "guid": "{b453ae62-4e3d-5e58-b989-0a998ec441b8}",
                "hidden": false,
                "name": "Azure Cloud Shell",
                "source": "Windows.Terminal.Azure"
            },
            {
                "colorScheme": "purplepeter",
                "icon": "D:/assets/img/png/icons8-gpu1.png",
                "guid": "{038fddf4-e674-4222-996b-02f81df69d2c}",
                "hidden": false,
                "name": "GPU-Lab 190",
                "commandline": "ssh kevin@172.31.224.190"
            },
            {
                // "background": "#013456",
                "acrylicOpacity": 0.8,
                "useAcrylic": true,
                "colorScheme": "purplepeter",
                "icon": "D:/assets/img/png/icons8-gpu1.png",
                "guid": "{8052abb9-733e-47ce-950f-b42a74445d72}",
                "hidden": false,
                "name": "GPU-Lab 142",
                "commandline": "ssh rpcv@172.31.233.142"
            },
            {
                "icon": "D:/assets/img/png/icons8-router.png",
                "guid": "{353365a7-d400-46aa-a429-987088d29576}",
                "hidden": false,
                "name": "Router",
                "commandline": "ssh root@192.168.1.1"
            },
            {
                "icon": "D:/assets/img/png/icons8-cloud.png",
                "guid": "{89c1c262-5b11-44f5-b186-0aa4a0c36567}",
                "hidden": false,
                "name": "Alibaba Cloud",
                "commandline": "ssh root@cloud.szukevin.site"
            }
        ]
    },

    // Add custom color schemes to this array.
    // To learn more about color schemes, visit https://aka.ms/terminal-color-schemes
    "schemes": [
        {
            "name": "purplepeter",
            "black": "#0a0520",
            "red": "#ff796d",
            "green": "#99b481",
            "yellow": "#efdfac",
            "blue": "#66d9ef",
            "purple": "#e78fcd",
            "cyan": "#ba8cff",
            "white": "#ffba81",
            "brightBlack": "#100b23",
            "brightRed": "#f99f92",
            "brightGreen": "#b4be8f",
            "brightYellow": "#f2e9bf",
            "brightBlue": "#79daed",
            "brightPurple": "#ba91d4",
            "brightCyan": "#a0a0d6",
            "brightWhite": "#b9aed3",
            "background": "#2a1a4a",
            "foreground": "#ece7fa"
          },
          {
            "name": "Ubuntu",
            "black": "#2e3436",
            "red": "#cc0000",
            "green": "#4e9a06",
            "yellow": "#c4a000",
            "blue": "#3465a4",
            "purple": "#75507b",
            "cyan": "#06989a",
            "white": "#d3d7cf",
            "brightBlack": "#555753",
            "brightRed": "#ef2929",
            "brightGreen": "#8ae234",
            "brightYellow": "#fce94f",
            "brightBlue": "#729fcf",
            "brightPurple": "#ad7fa8",
            "brightCyan": "#34e2e2",
            "brightWhite": "#eeeeec",
            "background": "#300a24",
            "foreground": "#eeeeec"
          }
    ],

    // Add custom keybindings to this array.
    // To unbind a key combination from your defaults.json, set the command to "unbound".
    // To learn more about keybindings, visit https://aka.ms/terminal-keybindings
    "keybindings":
    [
        // Copy and paste are bound to Ctrl+Shift+C and Ctrl+Shift+V in your defaults.json.
        // These two lines additionally bind them to Ctrl+C and Ctrl+V.
        // To learn more about selection, visit https://aka.ms/terminal-selection
        { "command": {"action": "copy", "singleLine": false }, "keys": "ctrl+c" },
        { "command": "paste", "keys": "ctrl+v" },

        // Press Ctrl+Shift+F to open the search box
        { "command": "find", "keys": "ctrl+shift+f" },

        // Press Alt+Shift+D to open a new pane.
        // - "split": "auto" makes this pane open in the direction that provides the most surface area.
        // - "splitMode": "duplicate" makes the new pane use the focused pane's profile.
        // To learn more about panes, visit https://aka.ms/terminal-panes
        { "command": { "action": "splitPane", "split": "auto", "splitMode": "duplicate" }, "keys": "alt+shift+d" }
    ]
}

```



对了，图标可以去 icons8 和 iconfont 下载哦，我都是前面这个网站下载的 96 px 的 icon，很好看，最后分享一波我的配置图，颜值第一生产力没错了

![mine](https://i.loli.net/2020/09/05/Il2RqswzQ8JUbWO.png)



## reference



https://juejin.im/post/6844904116322304014

https://sspai.com/post/59380

