# 2.1if
a = True  # bool型
# =赋值
# ==判断
if not a:  # 等价于a == False
    print("yes")
if a:  # 等价于a == True
    print("no")
# 双分支
b = True
if b:
    print(b)
else:
    print("error")
# 多分支
t = 5
if t == 1:
    print("星期一")
elif t == 2:
    print("星期二")
elif t == 3:
    print("星期三")
else:
    print("error!")
# ==
# >
# >=
# <
# <=
# != bool型可用not

a = '文件1'
b = '文件2'
print(a == b)

print(2 < 3 and 2 < 5)
print(2 > 3 or 3 == 3)  # 仅需满足一个条件即可
print(2 > 3 or not 3 == 3 and 5 < 10)
