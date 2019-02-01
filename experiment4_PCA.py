import struct
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


feature_num = 28 #特征的数量，二维数据测试就设为2，三维数据测试就设为3，mnist数据测试就设为28
expected_feature_num = 4 #期望降低到的维度
data_num = 28 #数据的数量，如果是mnist数据测试就必须设为28，其他无限制
u_3d = [2, 10, 4] #三维数据生成的均值矩阵
cov_3d = [[2, 2, 2], [2, 10, 2], [2, 2, 10]] #三维数据生成时的协方差矩阵
u_2d = [2,10] #二维数据生成的均值矩阵
cov_2d = [[10,9],[9,10]] #二维数据生成时的协方差矩阵

#获得矩阵a的第n列的方法
def getColumn(a, n):
    column = []
    for i in range(len(a)):
        column.append(a[i][n])
    return column

#返回一个矩阵中前n大的元素的索引
def find_max_n(data, n):
    min = -100000
    copy_data = []
    for i in range(len(data)):
        copy_data.append(data[i])
    temp = []
    for i in range(n):
        temp.append(np.argmax(copy_data))
        copy_data[int(np.argmax(copy_data))] = min
    return temp

#PCA算法，返回降维以后的数据和其对应的原坐标系下的坐标
def PCA(data):
    #首先对数据进行中心化处理
    mean = np.zeros(feature_num)
    for i in range(feature_num):
        mean[i] = sum(getColumn(data,i)) / data_num
    remove_data = []
    for i in range(data_num):
        line = []
        for j in range(feature_num):
            line.append(data[i][j] - mean[j])
        remove_data.append(line)
    #计算协方差矩阵
    cov = np.cov(np.array(remove_data).T)
    #计算特征值和特征向量
    eigenvalues, feature_vector = np.linalg.eig(cov)
    #获取特征值前expected_feature_num个大对应的特征向量
    max_n_eigenvalues = find_max_n(eigenvalues,expected_feature_num)
    result_vectors = []
    for i in range(len(max_n_eigenvalues)):
        result_vectors.append(getColumn(feature_vector,max_n_eigenvalues[i]))

    lower_data = np.dot(remove_data,np.array(result_vectors).T) #获得降维以后的数据
    result_point = np.dot(lower_data, result_vectors) #获得降维以后的数据对应原坐标系下的坐标
    #去中心化，将数据平移到原来的位置
    for i in range(data_num):
        for j in range(feature_num):
            result_point[i][j] += mean[j]
    return lower_data, result_point

#从文件中读取出mnist数据集
def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        buf = f.read()

    offset = 0
    magic, imageNum, rows, cols = struct.unpack_from('>IIII', buf, offset)
    offset += struct.calcsize('>IIII')
    images = np.empty((imageNum,rows, cols))
    image_size = rows * cols
    fmt = '>' + str(image_size) + 'B'

    for i in range(imageNum):
        images[i] = np.array(struct.unpack_from(fmt, buf, offset)).reshape((rows,cols))
        offset += struct.calcsize(fmt)
    return images

#计算两个图片矩阵对应的峰值信噪比
def psnr(img1, img2):
    difference = np.abs(img1 - img2)
    r = np.sqrt(difference).sum()
    p = 20 * np.log10(255/r)
    return  p

####二维向量的测试，使用时将对应的代码去掉注释，并将三维和mnist的测试注释掉
# data_2d = np.random.multivariate_normal(u_2d, cov_2d, data_num)
# lower_data, result_data = PCA(data_2d)
# plt.plot(getColumn(data_2d,0),getColumn(data_2d,1),"o")
# plt.plot(getColumn(result_data,0),getColumn(result_data,1),"o")
# plt.show()

####三维向量的测试，使用时将对应的代码去掉注释，并将二维和mnist的测试注释掉
# data_3d = np.random.multivariate_normal(u_3d, cov_3d, data_num)
# lower_data, result_data = PCA(data_3d)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(getColumn(data_3d,0), getColumn(data_3d,1), getColumn(data_3d,2),"o")
# ax.scatter(getColumn(result_data,0), getColumn(result_data,1),getColumn(result_data,2),"v")
# plt.savefig('fig.png',bbox_inches='tight')
# plt.show()

####mnist数据的测试，使用时将对应的矩阵去掉注释，并将二维和三维的测试注释掉
##数据的提取
array = decode_idx3_ubyte("train-images.idx3-ubyte")
data_mnist = array[0]
plt.subplot(1,2,1)
plt.title("Before dimension reduction")
plt.imshow(data_mnist,cmap='Greys',interpolation="nearest")
##计算并展示
lower_data_mnist, result_data_mnist = PCA(data_mnist)
plt.subplot(1,2,2)
plt.title("After dimension reduction")
plt.imshow(result_data_mnist,cmap='Greys',interpolation="nearest")
print(psnr(data_mnist, result_data_mnist)) #打印出信噪比
plt.show()
