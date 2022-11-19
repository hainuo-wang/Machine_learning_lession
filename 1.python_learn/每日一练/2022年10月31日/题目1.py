# 【问题描述】
#
# 求解圆周率可以采用蒙特卡罗方法，在一个正方形中撒点，根据在 1/4 圆内点的数量占总撒点数的比例计算圆周率值。
#
# 请以 123 作为随机数种子，获得用户输入的撒点数量，编写程序输出圆周率的值，保留小数点后 6 位。
#
# 【输入形式】
# 【输出形式】
# 【样例输入】
#
# 输入："1024"
#
# 【样例输出】
#
# 输出："3.218750"
#
# 【样例说明】
# 【评分标准】


import random

random.seed(123)
r = 1
numbers = eval(input())
ss = 0
for i in range(numbers):
    x, y = random.random(), random.random()
    ra = pow((x ** 2 + y ** 2), 0.5)
    if ra <= 1:
        ss += 1
pi = (ss / numbers) * 4
print("{:.6f}".format(pi))


