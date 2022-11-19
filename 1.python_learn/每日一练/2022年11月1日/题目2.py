# 题目2:
# 【问题描述】
#
# 利用可变参数，输出从键盘输入的数值的顺序排列的列表和他们的乘积。

def cmul(*b):
    lst = []
    a = 1
    for i in b:
        a *= i
        lst.append(i)
    lst.sort()
    return lst, a


p = eval(input())  # 以逗号分隔
lst, a = cmul(*p)
print(lst)
print(a)
