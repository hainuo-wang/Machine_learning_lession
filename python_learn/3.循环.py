# 1.for循环 有限的，范围，序列
for i in range(5):
    print(i)
for a in range(0, 5):  # 左闭右开
    print(a)
for b in range(0, 5, 2):
    print(b)
for c in range(5, 0, -1):
    print(c)
# 2.while 无限的，带有判断的
m = 0
while True:
    m += 1
    print("hello")
    if m == 10:
        break

a = 10
b = 0
while a > b:
    b += 2
    a += 1
    print(a, b)
# continue break
