---
layout: post
title: MMCV之Config注释详解
subtitle: 
date: 2023-05-05
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - object detection
    - deep learning
    - mmcv
---

# 前言

对 MMCV Config 类的结构记录一下，这个类主要是将 python dict 或者 json/yaml 文件中的 dict 对象转化成方便操作的 dict 对象，有些细节写的还是很好的，本文档用的 MMCV 的版本为 1.3.5

# class ConfigDict

这个类别继承了 addict 中的 Dict 类，可以通过访问属性的方式来访问字典中的值，其中重写了 `__missing__` 和 `__getattr__` 这两个魔法函数，因为对于 addict 中的 Dict，当字典中不存在 key 时会调用 `__missing__` 方法返回一个空的字典，而对于 ConfigDict ,当字典中不存在 key 时会直接报错，而不是返回一个默认值。

```Python
class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex
```

其中，addict 对 python 默认 dict 的加强在于重写了 `__getattr__` 和 `__setattr__` 函数，这两个函数让用户可以通过访问属性的方式（也就是 `a.b`）来访问字典中的值，不过 addict 可以嵌套多层，比较强大，我们可以重写这两个函数来实现一个简单的 demo：

```Python
class MyDict(dict):

    def __setattr__(self, __name: str, __value) -> None:
        print('__setattr__')
        self[__name] = __value

    def __getattr__(self, item):
        print('__getattr__')
        return self[item]

md = MyDict()
md.a = 1
print(md.a)
# setattr
# getattr
# 1
```

不过上述的 demo 还没办法做到嵌套调用，mmcv 官方写的这个最简版本 demo 可以实现嵌套调用，本质上就是对 dict 的值进行深度遍历

```Python
class MiniDict(dict):
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            for key, val in arg.items():
                # 对字典对象进行属性设置，并进行迭代
                self[key] = self._hook(val)
    def _hook(self, item):
        if isinstance(item, dict):
            return MiniDict(item)  
        return item
    # 递归调用return item# 在.a和['a']时候自动调用
    def __getattr__(self, item):
        return self[item]
        
r = MiniDict(dict(a=dict(b=2)))
print(r.a.b)
# 2
```

# class Config

## __init__

初始化函数，一般不会直接创建一个 Config 对象，而是从文件中读取 dict 以及其他信息作为参数传入初始化函数中，返回一个 Config 对象

```Python
def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
    if cfg_dict is None:
        cfg_dict = dict()
    elif not isinstance(cfg_dict, dict):
        raise TypeError('cfg_dict must be a dict, but '
                        f'got {type(cfg_dict)}')
    # 传进来的 dict 里面不能有预留的 key，不然报错
    for key in cfg_dict:
        if key in RESERVED_KEYS:
            raise KeyError(f'{key} is reserved for config file')

    # Config 没有显式的父类，所以继承了 Object 这个类
    # 调用父类的方法是因为 Config 重写了 __setattr__ 和 __getattr__ 函数
    # 所以要用父类的方法，不然就会陷入死循环
    super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
    super(Config, self).__setattr__('_filename', filename)
    if cfg_text:
        text = cfg_text
    elif filename:
        with open(filename, 'r') as f:
            text = f.read()
    else:
        text = ''
    super(Config, self).__setattr__('_text', text)
```

## fromfile 

这个是最重要的函数，也就是从文件中读取 config，然后变成 Config 对象。由于是静态函数，所以可以不通过对象来调用，可以直接通过 Config 类调用，也就是 `Config.fromfile`

```Python
@staticmethod
def fromfile(filename,
             use_predefined_variables=True,
             import_custom_modules=True):
    cfg_dict, cfg_text = Config._file2dict(filename,
                                           use_predefined_variables)
    # 可以在这里灵活导入一些自定义的模块
    if import_custom_modules and cfg_dict.get('custom_imports', None):
        import_modules_from_strings(**cfg_dict['custom_imports'])
    return Config(cfg_dict, cfg_text=cfg_text, filename=filename)
```

## _file2dict

fromfile 的主要逻辑，很重要！

```Python
@staticmethod
def _file2dict(filename, use_predefined_variables=True):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')
    
    # 这里创建了一个临时文件来保存原来的 config 文件，是为了让文件名可以出现 `a.b.py` 这种形式
    # 如果 config 是存储在 py 文件中的话，则是通过 import 来进行读取的，如果 import a.b，
    # 则会认为 a 是一个包的名字，就会出错，其实模块名叫 a.b，
    # 因此这里就巧妙地通过操作系统的 copy 将原文件换了个合理的名字保存在 tmp 文件夹中
    # 避免了导入模块时会发生的错误
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        # 替换一些 mmcv 预定义好的模版变量，默认是 True
        if use_predefined_variables:
            Config._substitute_predefined_vars(filename,
                                               temp_config_file.name)
        else:
            shutil.copyfile(filename, temp_config_file.name)

        if filename.endswith('.py'):
            temp_module_name = osp.splitext(temp_config_name)[0]
            # 将 temp_config_dir 添加到环境变量中，方便找到模块进行导入
            sys.path.insert(0, temp_config_dir)
            # 用 ast 抽象语法树检查 python 文件的格式
            Config._validate_py_syntax(filename)
            # 将存储着配置的 py 文件导入
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            # 只要是不带有 __ 开头的 key 全都保存在 cfg_dict 中
            # 
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            # 存储完之后就把这个模块给删了
            del sys.modules[temp_module_name]
        # 如果是其他后缀的文件的话就直接用 mmcv 导入成字典格式
        elif filename.endswith(('.yml', '.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(temp_config_file.name)
        # close temp file
        temp_config_file.close()

    cfg_text = filename + '\n'
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        cfg_text += f.read()
    
    # BASE_KEY 默认是 _base_，为继承的配置
    if BASE_KEY in cfg_dict:
        cfg_dir = osp.dirname(filename)
        # 获取到 base 文件名，用列表装，因为 base 文件可能有很多个
        base_filename = cfg_dict.pop(BASE_KEY)
        base_filename = base_filename if isinstance(
            base_filename, list) else [base_filename]

        cfg_dict_list = list()
        cfg_text_list = list()
        for f in base_filename:
            # 读取 base 文件中的配置，这边其实是个递归，就是 base 文件中也允许有 _base_ 字段
            _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
            cfg_dict_list.append(_cfg_dict)
            cfg_text_list.append(_cfg_text)

        base_cfg_dict = dict()
        for c in cfg_dict_list:
            # 不同的 base 文件中不允许存在相同的 key
            if len(base_cfg_dict.keys() & c.keys()) > 0:
                raise KeyError('Duplicate key is not allowed among bases')
            base_cfg_dict.update(c)
        
        # 将 base 文件中的配置合并到该文件的配置中
        base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = base_cfg_dict

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)
    
    # 所以 cfg_dict 就是合并之后处理完的配置，形式为 dict
    # cfg_text 就是字符串，包含了所有 base 文件中的内容以及该文件的配置内容
    return cfg_dict, cfg_text
```

## _substitute_predefined_vars

这个函数就是预定义了一些模版变量，在实际创建对象的时候将这些变量替换成用户独特的值。

```Python
@staticmethod
def _substitute_predefined_vars(filename, temp_config_name):
    # 这里获取到了文件的 4 种属性
    file_dirname = osp.dirname(filename)
    file_basename = osp.basename(filename)
    file_basename_no_extension = osp.splitext(file_basename)[0]
    file_extname = osp.splitext(filename)[1]
    # 支持下面这些属性
    support_templates = dict(
        fileDirname=file_dirname,
        fileBasename=file_basename,
        fileBasenameNoExtension=file_basename_no_extension,
        fileExtname=file_extname)
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        config_file = f.read()
    for key, value in support_templates.items():
        # 通过正则表达式将上面的 4 个模版替换成真实值
        # 正则表达式的意思是 {{ key }}，key 左右可以有0或0以上个空格
        regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
        value = value.replace('\\', '/')
        config_file = re.sub(regexp, value, config_file)
    # 这个上面分析过了，就是将原本的文件经过一些处理之后存在临时文件中，方便对 py 文件进行导入
    with open(temp_config_name, 'w') as tmp_config_file:
        tmp_config_file.write(config_file)
```

## import_modules_from_strings

这是 `mmcv.utils.misc` 中的一个函数，用来根据字符串导入 python 的模块。

```Python
def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported
```

## _merge_a_into_b

在 `_file2dict` 中，我们使用了 `Config._merge_a_into_b(cfg_dict, base_cfg_dict)` 将 base 文件中的配置和当前文件中的配置进行了合并，这里看看具体是怎么做的。

```Python
@staticmethod
def _merge_a_into_b(a, b, allow_list_keys=False):
    """merge dict ``a`` into dict ``b`` (non-inplace).

    Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
    in-place modifications.

    Args:
        a (dict): The source dict to be merged into ``b``.
        b (dict): The origin dict to be fetch keys from ``a``.
        allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
          are allowed in source ``a`` and will replace the element of the
          corresponding index in b if b is a list. Default: False.

    Returns:
        dict: The modified dict of ``b`` using ``a``.

    Examples:
        # Normally merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # Delete b first and merge a into b.
        >>> Config._merge_a_into_b(
        ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
        {'obj': {'a': 2}}

        # b is a list
        >>> Config._merge_a_into_b(
        ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
        [{'a': 2}, {'b': 2}]
    """
    b = b.copy()
    for k, v in a.items():
        # 允许列表作为 key，一般不会用这种情况，先不管
        if allow_list_keys and k.isdigit() and isinstance(b, list):
            k = int(k)
            if len(b) <= k:
                raise KeyError(f'Index {k} exceeds the length of list {b}')
            b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
        # 如果 base 文件和当前文件中都有相同的 key，且当前 key 的 value 不含 __delete__
        # 那就对这个 key 的 value 进行递归的 _merge_a_into_b
        elif isinstance(v,
                        dict) and k in b and not v.pop(DELETE_KEY, False):
            allowed_types = (dict, list) if allow_list_keys else dict
            if not isinstance(b[k], allowed_types):
                raise TypeError(
                    f'{k}={v} in child config cannot inherit from base '
                    f'because {k} is a dict in the child config but is of '
                    f'type {type(b[k])} in base config. You may set '
                    f'`{DELETE_KEY}=True` to ignore the base config')
            b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
        else:
            # 很巧妙，对 b 中不存在的 key 直接加上去
            # 如果有 __delete__ 的 key 也直接替换成 a 的
            # 而且上一个 if 已经将 __delete__ 弹出，此时，b 的 key 中已经不包含 __delete__ 了
            b[k] = v
    return b
```

## 魔法函数

Config 类对很多魔法函数都重写了，旨在通过 addict 对 python 的字典更加方便地访问

```Python
    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)
```

## Reference

https://zhuanlan.zhihu.com/p/346203167

https://blog.csdn.net/qq_38253797/article/details/121471389
