import random
import math
import matplotlib.pyplot as plt
import numpy as np

data_num = 50 #生成数据的个数
feature_num = 2 ##特征个数,这里如果是自己生成的数据就要取为2，如果是读取文件的数据就要取为9
u1 = [1, 2]         #第一维数据的u
u2 = [1, 2]        #第二维数据的u
sigema = [0.4, 0.6] #第一维数据和第二维数据的sigema
lamda = 0.001 #超参数
mean0 = [2, 8] #第一类数据两维的均值
mean1 = [3, 7] #第二类数据两维的均值
cov = [[0.5, 0.4], [0.4, 0.5]] #两类数据生成的协方差阵

#生成所需数据量的正态分布数据
def generateData(u, sigema):
    data = []
    for i in range(data_num):
        data.append(random.gauss(u, sigema))
    return data

#对于每个矩阵中的元素求logistic函数值
def logistic(x):
    # print(x)
    result = []
    for i in range(len(x)):
        # print("   ")
        # print(x[i])
        result.append(1.0/(1+math.exp(-1 *np.linalg.det(x[i]))))
    return np.mat(result).T

#对一个数求logistic函数值
def logisticNum(x):
    return 1.0/(1 + math.exp(-1 * x))

#梯度下降法求解
def gradientAscent(data, T):
    alpha = 0.001 #定义步长
    dataMatrix = np.mat(data)
    w = np.ones(feature_num+1) #w从全1列向量开始迭代
    w = np.mat(w).T
    # print(w)
    cycle = 2000
    for i in range(cycle):
        A = np.dot(dataMatrix, np.mat(w))
        # print(A)
        E = logistic(A) - np.mat(T).T
        # print(E)
        w = w - alpha * np.dot(dataMatrix.T, E) #迭代方程
        # print(w)
    return w

#梯度下降法加正则项
def gradientAscentWithReg(data, T):
    alpha = 0.001 #定义步长
    dataMatrix = np.mat(data)
    w = np.ones(feature_num+1)#w从全1列向量开始迭代
    w = np.mat(w).T
    cycle = 2000
    for i in range(cycle):
        A = np.dot(dataMatrix, np.mat(w))
        E = logistic(A) - np.mat(T).T
        w = w - alpha*lamda*w - alpha * np.dot(dataMatrix.T, E) #迭代方程，和梯度下降不同就在正则项
    return w

#牛顿法
def newton(data, T):
    dataMatrix = np.mat(data)
    w = np.ones(feature_num+1) #w从全1列向量开始迭代
    w = np.mat(w).T
    circle = 1000
    for k in range(circle):
        A = []
        for i in range(2 * data_num):
            line = []
            for j in range(2 * data_num):
                if (i == j):
                    line.append(np.linalg.det(np.dot(data[i], w))*(1-np.linalg.det(np.dot(data[i], w))))
                else:
                    line.append(0)
            A.append(line)
        AMatrix = np.mat(A)
        AA = np.dot(dataMatrix, np.mat(w))
        E = logistic(AA) - np.mat(T).T
        U = np.dot(dataMatrix.T, E)
        w = w + np.dot(np.mat(np.dot(np.dot(dataMatrix.T, AMatrix), dataMatrix)).I, U) #迭代方程
    return w

#牛顿法加正则项 和牛顿法的不同就在于迭代方程中添加了正则项
def newton_reg(data, T):
    dataMatrix = np.mat(data)
    w = np.ones(feature_num + 1)
    w = np.mat(w).T
    circle = 1000
    for k in range(circle):
        A = []
        for i in range(2 * data_num):
            line = []
            for j in range(2 * data_num):
                if (i == j):
                    line.append(np.linalg.det(np.dot(data[i], w)) * (1 - np.linalg.det(np.dot(data[i], w))))
                else:
                    line.append(0)
            A.append(line)
        AMatrix = np.mat(A)
        AA = np.dot(dataMatrix, np.mat(w))
        E = logistic(AA) - np.mat(T).T
        U = np.dot(dataMatrix.T, E)
        w = w + np.dot(np.mat(np.dot(np.dot(dataMatrix.T, AMatrix), dataMatrix)).I, U) - lamda * w
    return w

#从文件中读取数据
# f = open('data.txt','r')
# data = []
# T = []
# for i in f.readlines(): ##数据处理
#     data_line = []
#     data_line.append(1)
#     data_all = i.split(",")
#     for j in range(len(data_all)):
#         if j != len(data_all)-1:
#             data_line.append(float(data_all[j]))
#         else:
#             if data_all[j].replace("\n","") == 'O':
#                 T.append(0)
#             else:
#                 T.append(1)
#     data.append(data_line)

# #生成满足贝叶斯公式的数据
X10 = generateData(u1[0], sigema[0])
X20 = generateData(u2[0], sigema[1])
X11 = generateData(u1[1], sigema[0])
X21 = generateData(u2[1], sigema[1])
data = []
T = []
for i in range(data_num):
    line = []
    line.append(1)
    line.append(X10[i])
    line.append(X20[i])
    data.append(line)
for i in range(data_num):
    line = []
    line.append(1)
    line.append(X11[i])
    line.append(X21[i])
    data.append(line)
T.append([0]*data_num + [1]*data_num)

# #生成不满足贝叶斯假设的数据
# X10, X20 = np.random.multivariate_normal(mean0, cov, data_num).T
# # X11, X21 = np.random.multivariate_normal(mean1, cov, data_num).T
# # data = []
# # T = []
# # for i in range(data_num):
# #     line = []
# #     line.append(1)
# #     line.append(X10[i])
# #     line.append(X20[i])
# #     data.append(line)
# # for i in range(data_num):
# #     line = []
# #     line.append(1)
# #     line.append(X11[i])
# #     line.append(X21[i])
# #     data.append(line)
# # T.append([0]*data_num + [1]*data_num)

x = np.arange(0, 5, 0.01) #给x坐标分配范围
# weight_gradient = gradientAscent(data,T)
# print(weight_gradient)

#梯度下降不加正则项
weight_gradient = gradientAscent(data,T)
y_gradient = -(weight_gradient[1]/weight_gradient[2])*x-weight_gradient[0]/weight_gradient[2]
ax1 = plt.subplot(2, 2, 1)
plt.title("Gradient Descent")
plt.plot(X10,X20,'o')
plt.plot(X11,X21,'*')
plt.plot(x,y_gradient.T)

#梯度下降加正则项
weight_reg = gradientAscentWithReg(data,T)
y_reg = -(weight_reg[1]/weight_reg[2])*x-weight_reg[0]/weight_reg[2]
ax2 = plt.subplot(2, 2, 2)
plt.title("Gradient Descent & regular term")
plt.plot(X10,X20,'o')
plt.plot(X11,X21,'*')
plt.plot(x,y_reg.T)

#牛顿法不加正则项
weight_newton = newton(data, T)
y_newton = -(weight_newton[1]/weight_newton[2])*x-weight_newton[0]/weight_newton[2]
ax3 = plt.subplot(2, 2, 3)
plt.title("Newton's method")
plt.plot(X10,X20,'o')
plt.plot(X11,X21,'*')
plt.plot(x,y_newton.T)

#牛顿法加正则项
weight_newton_reg = newton_reg(data, T)
y_newton_reg = -(weight_newton_reg[1]/weight_newton_reg[2])*x-weight_newton_reg[0]/weight_newton_reg[2]
ax4 = plt.subplot(2, 2, 4)
plt.title("Newton's method & regular term")
plt.plot(X10,X20,'o')
plt.plot(X11,X21,'*')
plt.plot(x,y_newton_reg.T)

#满足贝叶斯假设的测试数据生成
X10_test = generateData(u1[0], sigema[0])
X20_test = generateData(u2[0], sigema[1])
X11_test = generateData(u1[1], sigema[0])
X21_test = generateData(u2[1], sigema[1])

#不满足贝叶斯假设的测试数据生成
# X10_test, X20_test = np.random.multivariate_normal(mean0, cov, data_num).T
# X11_test, X21_test = np.random.multivariate_normal(mean1, cov, data_num).T

test_data = []
test_T = []
for i in range(data_num):
    line = []
    line.append(1)
    line.append(X10_test[i])
    line.append(X20_test[i])
    test_data.append(line)
for i in range(data_num):
    line = []
    line.append(1)
    line.append(X11_test[i])
    line.append(X21_test[i])
    test_data.append(line)
test_T.append([0]*data_num + [1]*data_num)

#从文件中读取测试数据
# f = open('testdata.txt','r')
# test_data = []
# test_T = []
# for i in f.readlines():
#     data_line = []
#     data_line.append(1)
#     data_all = i.split(",")
#     for j in range(len(data_all)):
#         if j != len(data_all)-1:
#             data_line.append(float(data_all[j]))
#         else:
#             if data_all[j].replace("\n","") == 'O':
#                 test_T.append(0)
#             else:
#                 test_T.append(1)
#     test_data.append(data_line)

#计算生成数据的正确率
right_sum = 0.0
for i in range(len(test_data)):
    loglinear = np.dot(np.mat(test_data[i]), np.mat(weight_gradient)) #根据传参矩阵的不同来获得不同方法的正确率
    if (loglinear<=0 and test_T[0][i])==0 or (loglinear>=0 and test_T[0][i])==1:
        right_sum += 1
print(right_sum/len(test_data))

#计算从文件中读取的数据的正确率
# right_sum = 0.0
# for i in range(len(test_data)):
#     loglinear = np.dot(np.mat(test_data[i]), np.mat(weight_gradient))
#     if (loglinear<=0 and test_T[i])==0 or (loglinear>=0 and test_T[i])==1:
#         right_sum += 1
# print(right_sum/len(test_data))

plt.show()

