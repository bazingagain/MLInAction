# Created by leon at 07/11/2017

from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(inX, dataSet, labels, k):
    """ K 近邻算法
    :param inX: 输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量, 其元素数目和dataSet的行数相同
    :param k: 前k个
    :return:
    """
    dataSetSize = dataSet.shape[0] # 读取矩阵第一维度的长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #和各个向量计算向量差
    sqDiffMat = diffMat**2  # 举证点乘平方
    sqDistance = sqDiffMat.sum(axis=1) # 行 和, 即每行的所有列值相加,最后行数不变,列数变为1
    distances = sqDistance**0.5 # 对矩阵的每个元素开根号, 最后的数值越小,说明距离越近
    sortedDistancies = distances.argsort() #返回数组值从小到大的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistancies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #根据第2个域进行降序排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLine = len(arrayOLines) #得到文件行数
    returnMat = zeros((numberOfLine, 3)) #创建返回的NumPy矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector


# group, labels = createDataSet()
# print(classify([0, 0], group, labels, 3))

datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
print(datingDataMat)
# print(datingLabels[0:20])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()


