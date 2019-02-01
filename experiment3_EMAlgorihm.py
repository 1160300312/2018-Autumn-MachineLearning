import random
import math
import matplotlib.pyplot as plt
import numpy as np
import operator

data_num = 50     #每一个簇中的样本数量，当要使用文件中读取数据时，要设置为210
cluster_num = 3   #簇的数量
feature_num = 2   #特征的数量，要使用文件中读取数据时，要设置为7
u1 = [1,4,3]      #三类数据的第一维均值
u2 = [3,1,5]      #三类数据的第二维均值
sigema = [0.7, 0.7 ,0.7]  #三类数据的方差


#计算两个向量之间的欧式距离
def countDistance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(sum)

#计算一个列表所有元素的平均值
def countAvg(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    return sum/len(list)

#得到a矩阵的第n个列向量
def getColumn(a, n):
    column = []
    for i in range(len(a)):
        column.append(a[i][n])
    return column

#使用k-means方法来聚类
def kmeans(data):
    mid = [] #初始化均值集合
    cluster1 = [] #初始化第一个簇
    cluster2 = [] #初始化第二个簇
    cluster3 = [] #初始化第三个簇
    #随机从所有样本中选取三个点作为均值向量
    random_list = random.sample(range(0, len(data)-1), cluster_num)
    for i in range(cluster_num):
        mid.append(data[random_list[i]])
    while True:
        mid_before = []
        for i in range(len(mid)): #记录这次迭代前的均值
            mid_before.append(mid[i])
        cluster1 = []
        cluster2 = []
        cluster3 = []
        for i in range(len(data)-1): #计算均值到每一个样本的距离，然后将样本归到离它最近的均值代表的簇中（E步）
            distance1 = countDistance(data[i], mid[0])
            distance2 = countDistance(data[i], mid[1])
            distance3 = countDistance(data[i], mid[2])
            if distance1<distance2 and distance1<distance3:
                cluster1.append(data[i])
            if distance2<distance1 and distance2<distance3:
                cluster2.append(data[i])
            if distance3<distance1 and distance3<distance2:
                cluster3.append(data[i])
        #重新计算均值（M步）
        a_mid = []
        for j in range(feature_num):
            a_mid.append(countAvg(getColumn(cluster1,j)))
        mid[0] = a_mid
        a_mid = []
        for j in range(feature_num):
            a_mid.append(countAvg(getColumn(cluster2,j)))
        mid[1] = a_mid
        a_mid = []
        for j in range(feature_num):
            a_mid.append(countAvg(getColumn(cluster3,j)))
        mid[2] = a_mid
        if operator.eq(mid,mid_before):
            break
    return mid,cluster1,cluster2,cluster3

#计算一个多维高斯分布的值
def gaussion(data, u, cov):
    cov_det = np.linalg.det(cov)
    if cov_det == 0:
        cov_det = np.linalg.det(cov +np.eye(feature_num)*0.01)
        cov_I = np.linalg.inv(cov +np.eye(feature_num)*0.01)
    else:
        cov_I = np.linalg.inv(cov)
    index = -0.5 * np.dot(np.dot(np.array(data)-np.array(u), cov_I), (np.array(data)-np.array(u)).T)
    result = 1.0 / ((2*math.pi)**math.sqrt(cov_det)) * np.exp(index)
    return result

#计算似然值，用于分析
def count_likehood(data, u, sigema, alpha):
    result = 1
    for i in range(len(data)):
        sum = 0
        for j in range(cluster_num):
            sum += alpha[j]*gaussion(data[i],u[j],sigema[j])
        result *= sum
    return math.log(result)

#使用混合高斯模型+EM估计参数来聚类
def GMM(data):
    # likehood = []
    u = [] #初始化均值
    random_list = random.sample(range(0, len(data) - 1), cluster_num) #随机选取样本作为均值
    for i in range(cluster_num):
        u.append(data[random_list[i]])
        # print(u)
    cov = [] #初始化协方差阵
    for i in range(cluster_num):
        cov.append(np.cov(np.array(data).T))
    alpha = [] #初始化alpha
    for i in range(cluster_num):
        alpha.append(1.0 / cluster_num)
    gamma = [] #初始化gamma矩阵
    for i in range(len(data)):
        gamma.append(np.ones(cluster_num))
    circle = 10 #迭代次数
    for i in range(circle):
        # print(u)

        #E步 更新gamma矩阵
        for k in range(len(data)):
            sum = 0
            for j in range(cluster_num):
                sum += alpha[j] * gaussion(data[k], u[j], cov[j])
            for j in range(cluster_num):
                gamma[k][j] = alpha[j] * gaussion(data[k], u[j], cov[j]) / sum

        #M步 求参数的似然
        for j in range(cluster_num):
            sum1 = 0.0
            sum2 = 0.0
            for k in range(len(data)):
                sum1 += np.dot(gamma[k][j], data[k])
                sum2 += gamma[k][j]
            u[j] = sum1 / sum2
            sum_cov = 0
            for k in range(len(data)):
                sum_cov += gamma[k][j] * np.dot(np.array([data[k]-u[j]]).T, [data[k]-u[j]])
            cov[j] = sum_cov / sum2
            alpha[j] = sum2 / (len(data))
        # likehood_line = []
        # likehood_line.append(i)
        # likehood_line.append(count_likehood(data, u, cov, alpha))
        # likehood.append(likehood_line)
    return u,cov,gamma

#计算正确率的函数
def count_right(index,i,j,k,gamma,expect):
    if gamma[index][0] >= gamma[index][1] and gamma[index][0] >= gamma[index][2] and expect[index] == i:
        return 1
    elif gamma[index][1] >= gamma[index][0] and gamma[index][1] >= gamma[index][2] and expect[index] == j:
        return 1
    elif gamma[index][2] >= gamma[index][0] and gamma[index][2] >= gamma[index][1] and expect[index] == k:
        return 1
    else:
        return 0

#得到a列表和b列表中相同元素的个数
def count_equal(a,b):
    sum = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                sum += 1
    return sum



#生成数据
data = []
for i in range(data_num):
    for j in range(cluster_num):
        line = []
        line.append(random.gauss(u1[j], sigema[j]))
        line.append(random.gauss(u2[j], sigema[j]))
        data.append(line)
# plt.plot(getColumn(data,0),getColumn(data,1),"+")

#k-means方法执行并绘图
mid,clu1,clu2,clu3 = kmeans(data)
plt.subplot(2,2,1)
plt.title("k-means")
plt.plot(getColumn(mid,0),getColumn(mid,1),"+")
plt.plot(getColumn(clu1,0),getColumn(clu1,1),"o")
plt.plot(getColumn(clu2,0),getColumn(clu2,1),"v")
plt.plot(getColumn(clu3,0),getColumn(clu3,1),"1")

#混合高斯模型的执行和作图
cl1 = []
cl2 = []
cl3 = []
u, cov, gamma = GMM(data)
#画出似然值的折线图
# print(like_hood)
# plt.plot(getColumn(like_hood,0),getColumn(like_hood,1))
# plt.plot(getColumn(like_hood,0),getColumn(like_hood,1),"+")
for i in range(data_num*cluster_num):
    max = 0
    for j in range(cluster_num):
        if gamma[i][j] > gamma[i][max] :
            max = j
    if max == 0:
        cl1.append(data[i])
    if max == 1:
        cl2.append(data[i])
    if max == 2:
        cl3.append(data[i])
plt.subplot(2,2,2)
plt.title("Mixture of Gaussions")
plt.plot(getColumn(u,0),getColumn(u,1),"+")
plt.plot(getColumn(cl1,0),getColumn(cl1,1),"o")
plt.plot(getColumn(cl2,0),getColumn(cl2,1),"v")
plt.plot(getColumn(cl3,0),getColumn(cl3,1),"1")

##从文件中读取UCI的数据
# f = open('experiment3_data.txt','r')
# file_data = []
# expect_result = []
# for i in f.readlines():
#     line = []
#     for j in range(len(i.split())-1):
#         line.append(float(i.split()[j]))
#     expect_result.append(int(i.split()[len(i.split())-1]))
#     file_data.append(line)
# print(file_data)
#
##从文件中读取数据执行混合高斯模型
# file_u, file_cov, file_gamma = GMM(file_data)
#
##计算从文件中读取UCI的数据的混合高斯模型聚类的正确率
# right_sum = np.zeros(6)
# for i in range(len(file_gamma)):
#     right_sum[0] += count_right(i, 1, 2, 3, file_gamma, expect_result)
#     right_sum[1] += count_right(i, 1, 3, 2, file_gamma, expect_result)
#     right_sum[2] += count_right(i, 2, 1, 3, file_gamma, expect_result)
#     right_sum[3] += count_right(i, 2, 3, 1, file_gamma, expect_result)
#     right_sum[4] += count_right(i, 3, 1, 2, file_gamma, expect_result)
#     right_sum[5] += count_right(i, 3, 2, 1, file_gamma, expect_result)
# for i in range(6):
#     right_sum[i] /= data_num
# print(max(right_sum))
#
##从文件中读取数据执行k-means
# file_u_kmean, file_clu1, file_clu2, file_clu3 = kmeans(file_data)

##计算正确率
# file_right_clu1 = []
# file_right_clu2 = []
# file_right_clu3 = []
# for i in range(data_num):
#     if expect_result[i] == 1:
#         file_right_clu1.append(file_data[i])
#     if expect_result[i] == 2:
#         file_right_clu2.append(file_data[i])
#     if expect_result[i] == 3:
#         file_right_clu3.append(file_data[i])
# kmean_right_num = np.zeros(6)
# kmean_right_num[0] += count_equal(file_right_clu1,file_clu1) +  count_equal(file_right_clu2,file_clu2) +  count_equal(file_right_clu3,file_clu3)
# kmean_right_num[1] += count_equal(file_right_clu1,file_clu1) +  count_equal(file_right_clu2,file_clu3) +  count_equal(file_right_clu3,file_clu2)
# kmean_right_num[2] += count_equal(file_right_clu1,file_clu2) +  count_equal(file_right_clu2,file_clu1) +  count_equal(file_right_clu3,file_clu3)
# kmean_right_num[3] += count_equal(file_right_clu1,file_clu2) +  count_equal(file_right_clu2,file_clu3) +  count_equal(file_right_clu3,file_clu1)
# kmean_right_num[4] += count_equal(file_right_clu1,file_clu3) +  count_equal(file_right_clu2,file_clu1) +  count_equal(file_right_clu3,file_clu2)
# kmean_right_num[5] += count_equal(file_right_clu1,file_clu3) +  count_equal(file_right_clu2,file_clu2) +  count_equal(file_right_clu3,file_clu1)
# print(max(kmean_right_num)/data_num)

plt.show()