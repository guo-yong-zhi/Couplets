from main import *
print('输入上下联，或输入`q`退出')
print('如有下联用`|`隔开，下联空字用空格或减号占位')
print('输入例：\n白日依山尽\n白日依山尽|-河-海\n白日依山尽|明月\n')
while True:
    try:
        i = input("请输入：").split('|')
        first = i[0]
        second = i[1] if len(i)>1 else ''
        if first.startswith('q'): break
        print_match_all(first, second)
        print('='*8)
    except Exception as e:
        print(e)
