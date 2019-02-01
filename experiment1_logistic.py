import math
import random
import matplotlib.pyplot as plt
import numpy as np

n = 10           #测试数据数量
m = 9            #多项式阶数
lamda = 0.0001   #超参数
X = []           #生成的横坐标数组初始化
T = []           #生成的纵坐标数组初始化
u = 0            #高斯噪声的均值
sigema = 0.1     #高斯噪声的方差
gamma = 0.1      #梯度下降的步长

np.seterr(divide='ignore', invalid='ignore')

##构造系数为lambda的单位矩阵
identity_matrix = np.eye(m+1, dtype=int)
lamda_matrix = lamda * identity_matrix


##在[0,1]之间均匀生成n个数据点作为训练集
for num in range(0, n):
    x = num / n
    X.append(x)
    T.append(math.sin(2*math.pi*x))
for i in range(len(X)):
    T[i] += random.gauss(u, sigema)

plt.plot(X, T, linestyle='', marker='.') #将数据点在图中表示

##XX数组是对X矩阵的模拟
XX = []
for i in range(len(X)):
    XL = []
    for(j) in range(m+1):
        XL.append(X[i]**j)
    XX.append(XL)

##此处为带入解析解求矩阵的过程
XXM = np.matrix(XX) #X矩阵
XXTM = np.transpose(XXM) #X矩阵的转置
X_XT = np.dot(XXTM, XXM) #X矩阵的转置和X的乘积
XXTIM = np.matrix(X_XT).I #对上面的结果求逆
W = np.dot(np.matrix(np.dot(XXTIM, XXTM)), np.transpose(np.matrix(T))) #解析解W的求出

##这个方法传入的是W矩阵以及测试数据集合，返回的是loss值
def count_loss(w, data):
    loss_sum = 0
    for i in range(0,n):
        loss_sum += pow(count_data(w, data[i]) - math.sin(2*math.pi*data[i]), 2)
    return math.sqrt(loss_sum/n)

##这个方法是给定W矩阵和x值，得到多项式带入x值的结果
def count_data(w, x):
    function_sum = 0
    for i in range(0,m+1):
        function_sum += w[i] * pow(x,i)
    return function_sum

##梯度计算
def count_gradient(x, w, t):
    mid = np.dot(x, w) - t
    return 1./m * (np.dot(np.transpose(x), mid) + lamda * w)

##梯度下降法求出系数矩阵W
def countw(x, t):
    w = np.zeros((m+1,1), dtype=int) #初始化w为m+1*1的全0矩阵
    gradient = count_gradient(x, w, t) #计算梯度
    #循环直到梯度十分小
    while not np.all(np.absolute(gradient)<=1e-4) :
        w = w - gradient * gamma #往梯度方向下降一个步长
        gradient = count_gradient(x, w, t)
    return w

##使用共轭梯度法求解系数矩阵W
def conjugate_gradient(x, t):
    w = np.zeros((m+1, 1),dtype=int)#初始化w为m+1*1的全0矩阵
    r = t - np.dot(x, w)
    d = r
    ##迭代3n次 得到共轭梯度法的结果
    for i in range(3*n):
        alpha = np.dot(r.T, r) / np.dot(np.dot(d.T, x), d)
        w1 = w +  np.linalg.det(alpha) * d
        r1 = r - np.dot(np.dot(np.linalg.det(alpha), x),d)
        beta = np.dot(r1.T, r1)/ np.matrix(np.dot(r.T, r))
        d1 = r1 + np.dot(np.linalg.det(beta), d)
        d = d1
        w = w1
        r = r1
    return w

##定义画图过程中x的范围以及步长
x = np.arange(0, 1, 0.01)

##解析解
e = 0
for i in range(len(W)):
    e += W[i]*pow(x, i)
##得到函数并画出图像
ax3 = plt.subplot(2, 2, 1)
plt.plot(x, e.T)
plt.plot(X, T, linestyle='', marker='.')
plt.title("Analytical solution")
##限制横纵坐标的范围
plt.xlim(0, 1)
plt.ylim(-1, 1)

##解析解+惩罚项
##这里和解析解不同的就是添加了系数矩阵
##函数图像的得出和解析解相同
XXMR = np.matrix(XX)
XXTMR = np.transpose(XXMR)
X_XTR = np.dot(XXTMR, XXMR) + lamda_matrix  ##惩罚项的添加
XXTIMR = np.matrix(X_XTR).I
WR = np.dot(np.matrix(np.dot(XXTIMR, XXTMR)), np.transpose(np.matrix(T)))
er = 0
for i in range(len(WR)):
    er += WR[i]*pow(x, i)
ax4 = plt.subplot(2,2,2)
plt.plot(x, er.T)
plt.title("Analytic solution & regular term")
plt.plot(X, T, linestyle='', marker='.')
plt.xlim(0, 1)
plt.ylim(-1, 1)

#梯度下降法
w = countw(XX, np.transpose(np.matrix(T))) ##调用梯度下降的函数
E = 0
for i in range(len(w)):
    E += w[i]*pow(x, i)
ax2 = plt.subplot(2,2,3)
plt.plot(x, E.T)
plt.title("Gradient Descent")
plt.plot(X, T, linestyle='', marker='.')
plt.xlim(0, 1)
plt.ylim(-1, 1)

##共轭梯度法
w_conjugate = conjugate_gradient(X_XTR, np.dot(XXTM,  np.transpose(np.matrix(T)))) #调用共轭梯度的函数
e_conjugate = 0
for i in range(len(w_conjugate)):
    e_conjugate += w_conjugate[i]*pow(x, i)
ax1 = plt.subplot(2,2,4)
plt.plot(x, e_conjugate.T)
plt.title("Conjugate gradient")
plt.plot(X, T, linestyle='', marker='.')
plt.xlim(0, 1)
plt.ylim(-1, 1)

##实际函数
# y = np.sin(2 * math.pi * x)
# plt.plot(x, y)
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.show()

##生成测试数据并打印出loss
# sum = 0
# for j in range(5):
#     test_data = []
#     for i in range(0, n):
#         test_data.append(random.uniform(0, 1))
#     cost = count_loss(WR, test_data)
#     print('%.4f' % cost)
#     sum += cost
# print('%.4f' % (sum/5))

# test_data = []
# for i in range(0, n):
#     test_data.append(random.uniform(0, 1))
# cost1 = count_loss(W, test_data)
# print(cost1)
# cost2 = count_loss(WR,test_data)
# print(cost2)
# cost3 = count_loss(w, test_data)
# print(cost3)
# cost4 = count_loss(w_conjugate, test_data)
# print(cost4)

