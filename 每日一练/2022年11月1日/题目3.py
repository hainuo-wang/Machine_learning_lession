# 【问题描述】
#
# 请用递归编程实现。
#
# 求和 1!+2!+3！…+n！
#
# 【输入形式】
#
# 输入使用 input()，不要增加额外的提示信息
# 【输出形式】
#
# Python中 input 函数返回值是字符串, 可以使用 int(input()) 或 eval(input()) 来进行转换
#
# 【样例输入】"12"
#
# 【样例输出】"522956313"


def func(n):
    if n == 1:
        return 1
    else:
        return n * func(n - 1)


n = int(input())
count_sum = 0
for i in range(1, n + 1):
    count_sum = count_sum + func(i)
print(count_sum)

