import numpy

#这是一个变量类，用于封装numpy.ndarray类型的数据
class Variable:
    def __init__(self,data: numpy.ndarray):
        """
        :param
          data: numpy.ndarray类型的数据
        """
        self.data = data
        self.grad = None  #梯度初始化为None
        self.creator = None  #记录创建这个Variable的函数类实例
    def set_creator(self, function):
        """
        设置创建这个Variable的函数类实例
        :param
          function: Function类型的函数类实例
        """
        self.creator = function
    def backward(self):
        """反向传播函数
        计算梯度并更新Variable的grad属性
        """
        if self.grad is None:  #如果梯度为None，初始化为1
            self.grad = numpy.ones_like(self.data)
        #如果有创建者，递归调用前一个Variable的backward方法
        f = self.creator  #获取创建这个Variable的函数类实例
        if f is not None:
            x = f.input  #获取输入的Variable
            x.grad = f.backward(self.grad)  #调用函数类实例的backward方法计算梯度
            x.backward()  #递归调用前一个Variable的backward方法，直到没有创建者为止
        


# import numpy as np
# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# 这是一个函数类的基类
# 它的子类需要实现forward方法来定义具体的计算逻辑
# 这个类的实例可以被调用，输入一个Variable类型的参数，返回一个Variable类型的输出
class Function:
    def __call__(self, input: Variable) -> Variable:
        """
        :param
          input: Variable类型的输入
        :return 
            output: Variable类型的输出
        """
        x=input.data
        y=self.forward(x) #实际的计算
        output = Variable(y)#前向传播的结果封装成Variable类型
        output.set_creator(self) #设置创建这个Variable的函数类实例,这样会在前向传播的过程中就建立链接Define-by-Run
        #将输入的Variable变量保存到函数类实例中
        self.input = input #保存输入的Variable变量
        #将输出的Variable变量保存到函数类实例中
        self.output = output #保存输出的Variable变量
        #返回输出的Variable变量
        return output 
    def forward(self,x: numpy.ndarray) -> numpy.ndarray:
        """
        这个函数需要被子类实现
        :param
          x: numpy.ndarray类型的输入
        """
        raise NotImplementedError("forward函数未实现")
    def backward(self, grad_output: numpy.ndarray) -> numpy.ndarray:
        """
        反向传播函数，计算梯度
        :param
          grad_output: numpy.ndarray类型的梯度输出
        :return
          grad_input: numpy.ndarray类型的梯度输入
        """
        raise NotImplementedError("backward函数未实现")
    
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
    def backward(self, grad_output: numpy.ndarray) -> numpy.ndarray:
        x = self.input.data 
        #使用链式法则
        grad_input = 2 * x * grad_output
        return grad_input
    



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
    def backward(self, grad_output: numpy.ndarray) -> numpy.ndarray:
        x = self.input.data 
        #使用链式法则
        grad_input = numpy.exp(x) * grad_output
        return grad_input



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

# f=SquareFunction()
# x = numpy.array(2.0)
# # 计算数值微分
# grad = numerical_diff(f,x)
# print(grad)

A=SquareFunction()
B=ExpFunction()
C=SquareFunction()
x=Variable(numpy.array(0.5))
a=A(x)
b=B(a)
c=C(b)
c.backward()  #执行反向传播
print(x.grad)  #输出梯度


