import os
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
import types
import functools
import pickle
import json
import logging


class UFSet:
    '并查集'
    def __init__(self, eq_map_or_list=[]):
        '''
        eq_map_or_list: list or dict. [(o1,o2), (o3,o4,o5) ...] or {o1:o1, o4:o3, o5:o3}
        '''
        self.root = {}
        if isinstance(eq_map_or_list, dict):
            self.root = eq_map_or_list
        else:
            self.from_equivalence(eq_map_or_list)
        
    def find(self, u):
        if u not in self.root:
            return u
        else:
            _root = self.find(self.root[u]) #union管理的root中不可能存在环，递归必终止
            self.root[u] = _root #路径压缩
            return _root

    def union(self, u, v):
        '合并u、v。其中，保留u（的根），将v（的根）映射到u（的根）'
        ru = self.find(u)
        rv = self.find(v)
        if ru != rv:
            #两个元素的根不同，把它们并起来，把其中一个根成为另一个根的子节点
            self.root[rv] = ru

    def from_equivalence(self, es):
        'es: [(o1,o2), (o3,o4,o5) ...]'
        for ids in es:
            assert len(ids)>1
            for id_ in ids[1:]:
                self.union(ids[0], id_)
        return self

def cumulate(fun, seq, initial=None):
    '''
    返回累计施用fun的结果，类似np.cumsum的推广
    fun: 二元函数
    seq: 列表
    initial: 初始元
    return: list, 若initial不为None，长度等于seq；否则长度减一。
    '''
    if len(seq)==0:
        return seq
    else:
        if initial is not None:
            re = [initial]
            beg = 0
        else:
            re = [seq[0]]
            beg = 1
        for i in seq[beg:]:
            re.append(fun(re[-1], i))
    return re[1:]

iscollection = lambda v:(isinstance(v,Iterable)and(not isinstance(v,str)))

def dict_filter(cond, dic):
    return {k:v for k,v in dic.items() if cond(k,v)}

def dict_value_filter(cond, dic):
    return {k:v for k,v in dic.items() if cond(v)}

def dict_key_sort(old_dict, reverse=False, key=lambda x:x):
    items = sorted(old_dict.items(), key=lambda obj: key(obj[0]), reverse=reverse)
    new_dict = OrderedDict()
    for item in items:
        new_dict[item[0]] = old_dict[item[0]]
    return new_dict

def dict_value_sort(old_dict, reverse=False, key=lambda x:x):
    items = sorted(old_dict.items(), key=lambda obj: key(obj[1]), reverse=reverse)
    new_dict = OrderedDict()
    for item in items:
        new_dict[item[0]] = old_dict[item[0]]
    return new_dict

def dict_value_map(fun, *dicts):
    '''
    value map for dict
    >>> dict_value_map(lambda x:2*x, {1:2,3:4})
    {1: 4, 3: 8}
    >>> dict_value_map(lambda x,y:x+y, {1:2,3:4}, {1:5,3:6})
    {1: 7, 3: 10}
    '''
    keys = dicts[0].keys()
    for d in dicts[1:]:
        assert d.keys() == keys
    def dict_fun(*kvs):
        vs = (kv[1] for kv in kvs)
        return (kvs[0][0], fun(*vs))
    return dict(map(dict_fun, *(d.items() for d in dicts)))

def collection_map(fun, *collections):
    '''
    like standard map, but return corresponding collection. 
    support list, tuple, set, dict, generator, etc. 
    fun is applied to collection's value when collection is a dict.
    e.g.
    >>> collections_map(lambda x,y:x+y, [1,2], [3,4])  
    [4, 6]
    
    >>> collections_map(lambda x,y:x+y, {1:2,3:4}, {1:5,3:6})  
    {1: 7, 3: 10}
    
    >>> list(collections_map(lambda x:2*x, range(3)))  
    [0, 2, 4]
    '''
    assert len(collections) > 0
    if isinstance(collections[0], dict):
        return dict_value_map(fun, *collections)
    if isinstance(collections[0], (tuple, list, set)):
        return type(collections[0])(map(fun, *collections))
    else:
        return list(map(fun, *collections))

def apply(fun, *collections, at=lambda collections: not iscollection(collections[0])):
    '''
    like standard map, but can apply fun to inner elements.
    at: a int, a function or sometype. 
    at = 0 means fun(*collections)
    at = somefunction. fun will applied to the elements when somefunction(elements) is True
    at = sometype. fun will applied to the elements when elements are sometype.
    >>> apply(lambda x:2*x, [(1,2),(3,4)])  
    [(2, 4), (6, 8)]

    >>> apply(lambda a,b: a+b, ([1,2],[3,4]), ([5,6],[7,8]))
    ([6, 8], [10, 12])

    >>> apply(lambda a,b: a+b, ([1,2],[3,4]), ([5,6],[7,8]), at=1)  
    ([1, 2, 5, 6], [3, 4, 7, 8])

    >>> apply(lambda a,b: a+b, ([1,2],[3,4]), ([5,6],[7,8]), at=0)  
    ([1, 2], [3, 4], [5, 6], [7, 8])

    >>> apply(lambda a,b:a+b, {'m':[(1,2),[3,{4}]], 'n':5}, {'m':[(6,7),[8,{9}]],'n':10})  
    {'m': [(7, 9), [11, {13}]], 'n': 15}

    >>> apply(str.upper, [('a','b'),('c','d')], at=str)  
    [('A', 'B'), ('C', 'D')]
    '''
    if isinstance(at, int):
        assert at >= 0
        if at == 0:
            return fun(*collections)
        else:
            return collection_map(lambda *cs:apply(fun, *cs, at=at-1), *collections)
    if isinstance(at, types.FunctionType):
        if at(collections):
            return fun(*collections)
        else:
            return collection_map(lambda *cs:apply(fun, *cs, at=at), *collections)
    else:
        return apply(fun, *collections, at=lambda eles:isinstance(eles[0], at))

def map_reverse(map_dict, injective=True, iscollection=iscollection):
    """
    {obj1:[1,2], obj2:[2,3]} <=> {1:[obj1], 2:[obj1,obj2], 3:[obj2]}
    {obj1:1, obj2:1} => {1:[obj1,obj2]}
    {1:[obj1,obj2]} => {obj1:1, obj2:1} (injective=True)
    {1:[obj1,obj2]} => {obj1:[1], obj2:[1]} (injective=False)
    """
    new_dict = defaultdict(list)
    for k,v in map_dict.items():
        if iscollection(v):
            for vi in v:
                new_dict[vi].append(k)
        else:
            new_dict[v].append(k)
    if injective:
        new_dict2 = {}
        for k,v in new_dict.items():
            if len(v) <= 1:
                new_dict2[k] = v[0]
            else:
                new_dict2 = dict(new_dict)
                break
    else:
        new_dict2 = dict(new_dict)
    return new_dict2

def walk_leaves(dic, at=lambda ele: not isinstance(ele, dict), prefix=()):
    """
    dic: like '{'a':{'b':1,'c':2}}'
    at: bool function or a int indicates levels num to go down. 1 means {k:fun(v) for k,v in dic.items()}
    e.g.:
    >>> for pr, v in walk_leaves({'a':{'b':1,'c':2}, 'd':3}):  
    ...     print(pr, v)
        
    ('a', 'b') 1
    ('a', 'c') 2
    ('d',) 3
    """
    for k,v in dic.items():
        if isinstance(at, int):
            assert at >= 1
            if at == 1:
                yield (*prefix, k), v
            else:
                yield from walk_leaves(v, at-1, (*prefix, k))
        else:
            if at(v):
                yield (*prefix, k), v
            else:
                yield from walk_leaves(v, at, (*prefix, k))

# class nmmpy_io:
#     load = np.load
#     dump = lambda a,b,*args,**kargs: np.save(b, a, *args, **kargs)

def choose_serializer(f):
    if f.endswith('pkl'):
        serializer = pickle
    elif f.endswith('json'):
        serializer = json
    elif f.endswith('log'):
        serializer = iter_json
#     elif f.endswith('npy'):
#         serializer = nmmpy_io
    else:
        serializer = pickle
        print("选择plckle")
    return serializer

def with_cache(cache_files, function=None, overwrite=False, serializer='auto', open_mode='auto'):
    '''初次调用执行function返回结果，并将该结果序列化到cache_files；再次调用，直接读取cache_files返回结果
    cache_files: str or list of strs. if it's a list, len(cache_files) must be equal to len(function()).
    '''
    open_args_map = {pickle:'b', json:'t', iter_json:'t', nmmpy_io:'b'}
    if function is not None and not callable(function):
        function1 = lambda: function
    else:
        function1 = function
    if isinstance(cache_files, str):
        cache_files2 = [cache_files]
        function2 = lambda:[function1()]
    else:
        cache_files2 = cache_files
        function2 = function1

    loaded = []
    for f in cache_files2:
        if overwrite:
            break
        if not os.path.exists(f):
            print(f"未找到{f}，重新生成中...")
            break
        else:
            sl =  choose_serializer(f) if serializer == 'auto' else serializer
            om = open_args_map[sl] if open_mode == 'auto' else open_mode
            with open(f, 'r'+om) as file:
                if sl == iter_json:
                    r = sl.load(file, merge_dict=True)
                    if not isinstance(r, dict):
                        r = list(r)
                else:
                    r = sl.load(file)
                loaded.append(r)
    else:
        if isinstance(cache_files, str):
            assert len(loaded) == 1
            return loaded[0]
        else:
            return loaded
    
    results = function2()
    assert len(results) == len(cache_files2)
    for f,r in zip(cache_files2, results):
        sl =  choose_serializer(f) if serializer == 'auto' else serializer
        om = open_args_map[sl] if open_mode == 'auto' else open_mode
        with open(f, 'w'+om) as file:
            sl.dump(r, file)
    if isinstance(cache_files, str):
        assert len(results) == 1
        return results[0]
    else:
        return results


class lazyproperty:
    "惰性property描述符类，@lazyproperty类似@property"
    def __init__(self, fget):
        self.fget = fget
        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value


def corresponding_count(label1, label2):
    """
    label1, label2: list or dict. if dict, {id:label}
    return: like {label1_1:{label2_1:3,label2_2:4}...}
    """
    if not isinstance(label1, dict):
        label1 = dict(enumerate(label1))
    if not isinstance(label2, dict):
        label2 = dict(enumerate(label2))
    label_dict1 = map_reverse(label1)
    count_dict = {}
    for py, pxs in label_dict1.items():
        count_dict[py] = defaultdict(int)
        for px in pxs:
            if px in label2:
                count_dict[py][label2[px]] += 1
    return count_dict


def find_most_corresponding(label1, label2):
    """
    每个标签1对应的标签2是，标为标签1的所有objs的标签2的众数
    label1, label2: list or dict. if dict, {id:label}
    return: like {label1_1:label2_3, label1_2:label2_8}
    """
    cc = corresponding_count(label1, label2)
    return {k: max(v, key=v.__getitem__) for k, v in cc.items() if v}
