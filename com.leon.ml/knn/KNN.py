# Created by leon at 07/11/2017

from numpy import *
import operator
from os import listdir
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

def autoNorm(dataSet):
    """
    由于各特征值取值范围不同,但如果认为各特征值同等重要,则要对数据进行归一化,减少取值范围差异带来的影响
    newValue = (oldValue - min) / (max-min)
    其中,max,min分别为某列特征值中的最大值和最小值
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0) # 选取每列的最小值放入minVlas中
    maxVals = dataSet.max(0) # 选取每列的最大值放入maxVlas中
    ranges = maxVals - minVals # 得到max - min
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1)) # (oldValue - min)
    normDataSet = normDataSet / tile(ranges, (m, 1)) # (oldValue - min) / (max-min), 点除,非矩阵除法
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    测试,分类,计算错误率
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs): # 前0~numTestVecs行作为测试集,后numTestVecs~m行作为训练集
        classifierResult = classify(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%d, the real answer is:%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is:%f" % (errorCount/float(numTestVecs)))

def img2vector(filename):
    """
    将32 * 32的二进制图像转换为 1*1024的矩阵
    :param filename: 文件名
    :return:
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j]) # 第0行,第32*i+j个元素设置为 图像中的1或0
    return returnVect

def handWritingClassTest():
    """
    图片分类算法, 采用k-近邻相似处理
    :return:
    """
    hwLables = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLables, 5)
        print("the classifier came back with: %d, the real answer is:%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("the total number of error is:%d" % (errorCount))
    print("the total error rate is:%f" % (errorCount / float(mTest)))



# group, labels = createDataSet()
# print(classify([0, 0], group, labels, 3))

# datingClassTest()


"""

datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")

normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(normMat[:,0], normMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

"""

# handWritingClassTest()



