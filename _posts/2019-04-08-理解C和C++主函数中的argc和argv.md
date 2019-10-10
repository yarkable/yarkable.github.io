---
layout:     post
title:      理解C和C++主函数中的argc和argv
subtitle:   虽然这两个东西好像没什么用
date:       2019-04-08
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - C Programming
    - cpp
---

以前学 C 语言的时候用过一段时间老版的 Visual Studio，里面的 main 函数是这样定义的

```c++
int main(int argc, char *argv[])
```

当时老师就没给我们讲这是怎么回事，让我们这样用就行，甚至可以把它删了，我就不去管它了，反正不写也是对的，前几天偶然的机会看到一篇博客中说上面那种写法才是最正规的，我便去查阅资料，终于搞懂了这是什么意思。

|param|commit|
| :-- | :-- |
|int argc | argument count，命令行传入参数的个数|
|char *argv[] | argument vector，传入的参数向量或数组|

首先，如果我们定义了一个带参数的函数，调用它的时候就得传入实参才行，但是一个程序有且只有一个 main 函数，在程序内部我们是不可能传入参数的，只能在命令行运行生成的二进制文件（windows 下为 exe 后缀文件）才能在后面传入 main 函数的参数。

---
这里所说的命令行参数也就是类似 Linux 命令中的 `ls -l` 中的 `-l` 之类的东西

**举个例子**

```c++
#include "iostream"
using namespace std;
int main(int argc, char *argv[])
{
    cout << "arg number: " << argc << endl;
    cout << argv[0] << endl;
    system("sleep 1");
    return 0;
}
```

我们编译成功之后在命令行不带参数运行一下这个文件，会得到下面的结果(windows在命令行同理)

```
$ ./tests
```

```
arg number: 1
argv[0]: ./tests

```

其实这里的 `argc` 就是在命令行传入的参数的个数，你可能会说我没有传参数为什么 `argc` 会是 1 呢，其实第一个参数就是程序本身，像我们打印的 `argv[0]` 就是程序传入的第一个参数，他会将程序的完整路径给打印出来（这里是因为我在当前目录下运行文件），这根 Linux shell 脚本中命令行参数的 `$0` 指向的是脚本本身是一个道理。

再来看看 `char *argv[]`，我们知道 `char *arg` 可以表示一个字符串，那么简单来说 `char *argv[]` 就表示为一个字符串数组，所以他可以接受命令行中的那么多参数，那么我们再来举个例子，把上面的例子改一下

```c++
#include "iostream"
using namespace std;
int main(int argc, char *argv[])
{
    cout << "arg number: " << argc << endl;
    for (int i = 0; i < argc; i++){
        cout << "argv[" << i << "]: " << argv[i] << endl;
    }
    system("sleep 1");
    return 0;
}
```

这次我们在命令行输入

```
$ ./tests I am very happy
```

会得到下面的输出，将所有收到的参数全部对应输出了

```c++
arg number: 5
argv[0]: ./tests
argv[1]: I
argv[2]: am
argv[3]: very
argv[4]: happy

```

所以这两个参数其实就是拿来处理命令行参数用的，但是不一定要命名成 `argc` 和 `argv[]`，只要符合 C/C++ 命名规范就行，重要的是两个参数的类型，一个是 `int` 另一个是 `char **` 或 `char* []`。

网上查资料时说一般在命令行处理文件配合 stream 流多用这两个参数，反正我目前基本没用过，网上也没有比较经典的例子，但是还是了解一下比较好。

---

reference：

[[C/C++基础知识] main函数的参数argc和argv](https://blog.csdn.net/Eastmount/article/details/20413773)

[C++ main函数中参数argc和argv含义及用法](https://blog.csdn.net/dcrmg/article/details/51987413)

[深入 char * ,*char \*** ,*char a[ ] ,char *a[] 内核](https://blog.csdn.net/daiyutage/article/details/8604720)