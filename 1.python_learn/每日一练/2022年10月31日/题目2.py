# 【问题描述】统计一行字符的大写字母，小写字母和数字的个数。先输出大写字母个数，在输出小写字母个数，最后输出数字个数。
# 【输入形式】ljaij1A
# 【输出形式】1
#
#                         5
#
#                         1
# 【提示】用字符串的方法isupper, islower来判别大小写。isdigit来判断是否是数字。

s = input()
str_upper, str_lower, num = 0, 0, 0
for i in s:
    if i.islower():
        str_lower += 1
    elif i.isupper():
        str_upper += 1
    elif i.isdigit():
        num += 1
print(str_upper)
print(str_lower)
print(num)
