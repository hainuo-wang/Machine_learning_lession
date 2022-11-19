# 定义函数 无返回值
def modify_name(filename):
    """

    :param filename:
    :return:
    """
    filename += '.txt'
    print(filename)


# 调用函数
modify_name('王海诺')


# 定义函数
def modify_name1(filename):
    """

    :param filename:
    :return:
    """
    filename += '.txt'
    return filename


new_name = modify_name1('王海诺')
print(new_name)

# 参数可以有0-多个
# 参数可以是不定个数（可变参数）
