import numpy

#这是一个变量类，用于封装numpy.ndarray类型的数据
class Variable:
    def __init__(self,data: numpy.ndarray):
        """
        :param
          data: numpy.ndarray类型的数据
        """
        self.data = data


# import numpy as np
# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# 这是一个函数类的基类
# 它的子类需要实现forward方法来定义具体的计算逻辑
# 这个类的实例可以被调用，输入一个Variable类型的参数，返回一个Variable类型的输出
class Function:
    def __call__(self,input: Variable) -> Variable:
        """
        :param
          input: Variable类型的输入
        :return 
            output: Variable类型的输出
        """
        x=input.data
        y=self.forward(x) #实际的计算
        output = Variable(y)
        return output 
    def forward(self,x: numpy.ndarray) -> numpy.ndarray:
        """
        这个函数需要被子类实现
        :param
          x: numpy.ndarray类型的输入
        """
        raise NotImplementedError("forward函数未实现")
    
class SquareFunction(Function):
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        实现平方计算
        :param
          x: numpy.ndarray类型的输入
        :return 
          x的平方
        """
        return x ** 2

# x = Variable(numpy.array(5.0))
# f = SquareFunction()
# # 调用函数类的实例
# # 传入Variable类型的输入，返回Variable类型的输出
# y = f(x)
# print(type(y ))
# print(y.data) #输出计算后的结果

class ExpFunction(Function):
    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        实现指数计算
        :param
          x: numpy.ndarray类型的输入
        :return 
          e的x次方
        """
        return numpy.exp(x)


#数值微分
def numerical_diff(f: Function, x: numpy.ndarray, eps: float = 1e-4) -> numpy.ndarray:
    """
    数值微分函数
    :param
      f: Function类型的函数
      x: numpy.ndarray类型的输入
      eps: 微小值，用于计算导数
    :return
      导数的值
    """
    x0 = x - eps
    x1 = x + eps
    y0 = f(Variable(x0)).data
    y1 = f(Variable(x1)).data
    return (y1-y0) / (2 * eps)

f=SquareFunction()
x = numpy.array(2.0)
# 计算数值微分
grad = numerical_diff(f,x)
print(grad)
