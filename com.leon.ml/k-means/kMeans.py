# Created by leon at 09/11/2017

from numpy import *
import matplotlib.pyplot as plt
import queue
import threading
from multiprocessing.pool import ThreadPool

def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    indexList = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        indexList.append(curLine[0])
        curLine = curLine[1:-1]
        # fltLine = map(float, curLine)  # map all elements to float()
        i = 0
        for s in curLine:
            curLine[i] = float(s)
            i+=1
        dataMat.append(curLine)
    return dataMat, indexList


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 计算欧式距离


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # k 行,n列  随机创建k个初始簇心
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j])-minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] # 数据点数目
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k) # 得到随机创建的簇心, k行,n列
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :]) # 计算点到每个簇心的距离,找到最小距离的簇心的index
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2 # 第i个数据对应属于某个索引的簇心
        # print(centroids)
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 按列方向相加,并求平均值
    return centroids, clusterAssment

def testnormalKMeans():
    datMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 4)
    plt.plot(datMat[:,0].tolist(), datMat[:,1].tolist(), 'o', label='data')
    plt.plot(myCentroids[:,0].tolist(), myCentroids[:,1].tolist(), '+', label='cluster')
    plt.xlabel('input x')
    plt.ylabel('input y')
    plt.xlim([-6,6])
    plt.ylim([-6,6])
    plt.title(' K-means')
    plt.grid()
    plt.legend(loc=2)
    plt.show()

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2))) # m行2列，其中第一列存放质心的索引，第2列存放误差平方和(SSE)
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 计算所有点的唯一质心
    for j in range(m):  #  计算每个点的初始误差
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],:]  #获取第i个簇的数据点集合
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 对第i个簇进行2-Means聚类
            sseSplit = sum(splitClustAss[:, 1])  # 第i簇进行2-Means后的总误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1]) # 其它簇的总误差
            # print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 进行二分后的总误差与当前总误差比较
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 划分到第 len(centList) 新簇
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit # 划分到原第bestCentToSplit簇
        # print('the bestCentToSplit is: ', bestCentToSplit)
        # print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 将原来的簇心以两个新的簇心进行替代
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment

def testBiKMeans():
    datMat3 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments=biKmeans(datMat3, 3)
    plt.plot(datMat3[:,0].tolist(), datMat3[:,1].tolist(), 'o', label='data')
    plt.plot(centList[:,0].tolist(), centList[:,1].tolist(), '+', label='cluster')
    plt.xlabel('input x')
    plt.ylabel('input y')
    plt.xlim([-6,6])
    plt.ylim([-6,6])
    plt.title('bisecting K-means')
    plt.grid()
    plt.legend(loc=2)
    plt.show()

def testGoodK(k, q):
    datMat4, indexList = loadDataSet('testvec.txt')
    datMat4 = mat(datMat4)
    centerList, myNewAssments=biKmeans(datMat4, k)
    print(str(k) + ":" + str(shape(centerList)) + ":" + str(sum(myNewAssments[:,1])))
    # return sum(myNewAssments[:,1])
    q.put(sum(myNewAssments[:,1]))

def show():
    k = range(30,50)
    pool = ThreadPool(processes=5)
    qs = []
    errrate = []
    for i in k:
        q = queue.Queue()
        pool.apply_async(testGoodK, args=(i, q))
        qs.append(q)
    for q in qs:
        price = q.get()
        errrate.append(price)

    # plt.plot(k, [testGoodK(p) for p in k], 'o', label='data')
    plt.plot(k, errrate, 'o', label='data')
    plt.xlabel('input k')
    plt.ylabel('error Assement')
    plt.xlim([30, 50])
    plt.ylim([0, 20])
    plt.title('bisecting K-means')
    plt.grid()
    plt.legend(loc=2)
    plt.show()

show()

import urllib
import json


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print
    yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

# clusterClubs(5)