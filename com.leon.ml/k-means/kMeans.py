# Created by leon at 09/11/2017

from numpy import *
from math import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        tmp = []
        # fltLine = map(float, curLine)
        tmp.append(float(curLine[0]))
        tmp.append(float(curLine[1]))
        # dataMat.append(fltLine)
        dataMat.append(tmp)
    return dataMat

def disEclud(vecA, vecB):
    """
     计算两个向量的欧式距离
    :param vecA:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n))) # mat 将数组转化为矩阵
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


#  test
numMat = mat(loadDataSet('testSet.txt'))
print(min(numMat[:,1]))

