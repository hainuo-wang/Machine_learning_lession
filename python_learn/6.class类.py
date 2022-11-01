# 面向对象
# 声明
class File:
    # name = 'f1'  # 共有财产

    def __init__(self):
        self.name = 'f1'  # 私有财产
        self.create_time = 'today'

    def getdata(self):
        print(self.name)


# 实例化
file = File()
print(file.name)
file.getdata()

# 继承
