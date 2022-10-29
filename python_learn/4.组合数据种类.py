# 列表[]，字典{}，元组()，集合set{}
# 1.列表
lst = [1, 1.2, 'hello', (1, 2)]
print(lst)
lst.append('python')
print(lst)
lst.append('pytorch')
print(lst)
a = []
for i in range(0, 101, 2):
    if i % 2 == 0:
        a.append(i)
# for i in range(0, 101, 2):
#     a.append(i)
print(a)
# 切片
print(lst[0])
print(lst[2: 4])  # 左闭右开
print(lst[-2])
print(lst[-2:])

# 2.字典
# 键值对
# key:value
file = {'ID': '2020212833', 'name': '王海诺', 'class': '08032003'}
print(file)
print(file['name'])
# 4.元组 可读
tup = (2, 3, 4)
print(tup)
for i in tup:
    print(i)
# 4.集合 不重复性 用处：去重复值
lst_new = [1, 1, 1, 2, 3]
set_lst = set(lst_new)
print(set_lst)

# 组合数据类型相互之间也是可以组合的（套娃）
