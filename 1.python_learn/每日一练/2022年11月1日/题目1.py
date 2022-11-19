# 【问题描述】
#
# 编写程序， 输入一个大于 2 的自然数， 然后输出小于该数字的所有素数组成的列表。
# 【输入形式】
# 【输出形式】
# 【样例输入】
#
# 7
#
# 【样例输出】
#
# [2, 3, 5]


import math


def prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return 0
    else:
        return 1


num = int(input())
prime_num = list(range(2, num))
for i in prime_num:
    if prime(i) == 0:
        prime_num[i - 2] = -1
prime_num = set(prime_num)
prime_num = list(prime_num)
prime_num.remove(-1)
print(sorted(prime_num))

