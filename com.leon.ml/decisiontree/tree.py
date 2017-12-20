# Created by leon at 20/12/2017

from math import log
import operator
import treePlotter

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算数据集的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts ={}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    对数据集进行划分
    :param dataSet: 数据集 [[feat1value, feat2value, feat3value ..], [feat1value, feat2value, feat3value, ..]]
    :param axis: 指定要依据哪个特征进行划分
    :param value: 特质的值, 一个特征有多个值, 如size: small, normal, big
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatureVec = featVec[:axis]
            reduceFeatureVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatureVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    找到最佳的特征进行划分（决策）
    :param dataSet:
    :return: 特征在样本中的下标
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain =0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 得到某个特征的非唯一取值集合
        newEntropy = 0.0
        for value in uniqueVals: # 计算每个取值的信息熵,
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/ float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy # (信息增益)
        if (infoGain > bestInfoGain): # 得到最大的信息增益,即通过此种划分,信息混乱度减小幅度最大
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    由于存在处理了所有属性,但是依然存在类别不同的情况, 该方法通过多数表决的方式指定分类
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    构建决策树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet] # 获得分类集
    if classList.count(classList[0]) == len(classList): # 类别完全相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1: #遍历完所有特征时,返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals: # 特征的每一种值,都是一个树的分支
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filname):
    import pickle
    fr = open(filname, 'rb')
    return pickle.load(fr)



def test():
    mydat, labels = createDataSet()
    print(chooseBestFeatureToSplit(mydat))

def testCreateTree():
    mydat, labels = createDataSet()
    print(createTree(mydat, labels))

def test1():
    mydat, labels = createDataSet()
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    print(classify(myTree, labels, [1, 1]))

def testDumpAndLoadDecisionTree():
    myTree = treePlotter.retrieveTree(0)
    storeTree(myTree, 'classTree.txt')
    print(grabTree('classTree.txt'))

def testFromFile():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lesesTree = createTree(lenses, lensesLabels)
    print(lesesTree)
    treePlotter.createPlot(lesesTree)

testFromFile()