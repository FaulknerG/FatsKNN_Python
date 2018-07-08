from numpy import *
#from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score


def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    y = iris.target
    return X, y


# 计算欧几里得距离
def distEclud(arrA, arrB):
    return sqrt(sum(power(arrA - arrB, 2)))  # 求两个向量之间的距离


# 构建初始聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    # n = shape(dataSet[1])
    # n: 特征数
    n = dataSet.shape[1]
    centroids = mat(zeros((k, n)))
    # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def KMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
    # 样本数量
    m = dataSet.shape[0]
    # 样本所属类别以及到质心距离
    clusterAssment = mat(zeros((m,2)))
    # 生成初始化质心
    centroids = createCent(dataSet, k)
    # 标志位
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算距离，第i个点 与 第j个类别质心
                distJI = distMeans(dataSet[i, :], centroids[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        # print(centroids)
        # 重新计算质点
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 去第一列等于cent的所有列
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 算出这些数据的中心点
    return centroids, clusterAssment


if __name__ == '__main__':
    X, y = load_data()
    k = 3  # 类别数
    myCentroids, clustAssing = KMeans(X, k)

    print('============final centroids==============\n', myCentroids)



    # print(clustAssing)
    # km = KMeans(n_clusters=3,
    #             init='random',
    #             n_init=10,
    #             max_iter=300,
    #             tol=1e-04,
    #             random_state=0)
    # y_km = km.fit_predict(X)
    # y_pred = clustAssing.A[:, 0]
    # print('Misclassified samples: %d' % (y_pred != y).sum())
    # print(sum((y_pred-y) == 0))
    # print('Accuracy: %.2f' % accuracy_score(y, y_pred))




