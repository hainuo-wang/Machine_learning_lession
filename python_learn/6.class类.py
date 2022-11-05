# 面向对象
# 声明
class File:
    # name = 'f1'  # 类属性

    def __init__(self):  # 实例属性
        self.name = 'f1'
        self.create_time = 'today'

    def getdata(self):
        print(self.name)


# 实例化
file = File()
print(file.name)
file.getdata()

# 继承
