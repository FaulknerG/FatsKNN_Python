"""
==============聚类过程中的主要变量=============
l: 划分的子集数目
L: 水平数目
Xp: 节点p对应的样本子集
Mp: 各类均值
Rp: 从Mp到Xp的最远距离
==============树搜索过程中的主要变量===========
CurL: 当前水平
p: 当前节点
CurTable: 当前目录表中的子样本集
CurPinT: 在当前目标表中的子样本节点
RpCur: 当前目录表中节点p对应的Rp
x: 待判别样本
"""
from kmeans import PR_kmeans
from numpy import *
from sklearn.cluster import KMeans
from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    return X


def dist(vecA, vecB):
    # return sqrt(sum(power(vecA - vecB, 2)))
    # matlab中对距离不开方
    return sum(power(vecA - vecB, 2))


def PR_kmeans_gx(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    PR_IDX = kmeans.labels_
    PR_C = kmeans.cluster_centers_
    # 求 sumd : 所有该类距离中心的距离累加和
    PR_sumd = array([0.0, 0.0, 0.0])
    for centj in range(l):
        for nodei in range(data.shape[0]):
            if centj == PR_IDX[nodei]:
                PR_sumd[centj] += dist(data[nodei], PR_C[centj])
    # 求 D : 每个点距离三个中心点的距离。
    PR_D = zeros((data.shape[0], k))
    for nodei in range(data.shape[0]):
        for centj in range(l):
            PR_D[nodei, centj] = dist(data[nodei], PR_C[centj])
    return PR_IDX, PR_C, PR_sumd, PR_D


# 读取数据 iris数据
data, data_target = load_data()
row, col = data.shape
# ========================================================
# 首先进行聚类
nodeNum = 0
L = 3
l = 3
# 计算总节点数
for i in range(L):
    nodeNum += l**(i+1)

# 初始化变量
Xp = []  # 39 * 1
Mp = zeros((nodeNum, col))
Rp = zeros((nodeNum, 1))
p = 0

for i in range(1, 4, 1):
    if i == 1:
        IDX, C, sumd, D = PR_kmeans_gx(data, l)

        for j in range(l):
            Xp.append(data[(IDX == j), :])
            Mp[p, :] = C[j]
            Rp[p] = max(D[(IDX == j), j])
            p += 1
    else:
        # p : 3
        endk = p   # 3, 12
        begink = endk - l**(i-1) + 1  # i:1,2 | begink:1, 4
        for k in range(begink, endk+1, 1):   # 1-3, 4-12
            IDX, C, sumd, D = PR_kmeans_gx(Xp[k-1], l)
            X1 = Xp[k-1]
            for j in range(l):
                Xp.append(X1[(IDX == j), :])
                Mp[p, :] = C[j]
                Rp[p] = max(D[(IDX == j), j])
                p += 1

# ========================================================
# 进行树搜索
x = [6, 6, 6, 6]
B = 100000; CurL = 1; p = 0; TT = 1
while TT == 1:
    Xcurp = [[]]  # 1
    CurTable = []  # 3*1 当前目录表中的样本子集
    CurPinT = zeros((l, 1))  # 在当前目录表中的子样本节点
    Dx = zeros((l, 1))
    RpCur = zeros((l, 1))  # 当前目录表中节点p对应的Rp
    # 当前节点的直接后继放入目录表, 并对这些节点计算D(x,Mp)
    for i in range(l):         # i: 0-2
        Cur_p = i+p*l
        CurTable.append(Xp[Cur_p])
        CurPinT[i] = Cur_p
        Dx[i] = sum(pow((x - Mp[Cur_p]), 2))  # 距离,没有开根号
        RpCur[i] = Rp[Cur_p]

    while 1:
        rowT = CurTable.__len__()
        for i in range(rowT):
            # 按照规则1剪枝
            if Dx[i] > B + RpCur[i]:
                # 从当前目录表中去掉节点i
                del(CurTable[i])
                delete(CurPinT, i)
                delete(Dx, i)
                delete(RpCur, i)
                break
        CurRowT = CurTable.__len__()
        # 如果目录表中已经没有节点了,则后退一个水平
        if CurRowT == 0:
            CurL -= 1
            p = int((p-1)/3)
            # 如果L=0了，停止。======这是出口
            if CurL == 0:
                TT = 0
                break
            else:
                # 转步骤3，
                pass
        elif CurRowT > 0:
            p1 = Dx[:, 0].argmin()
            p = p1
            # 从当前目录表中去掉 p1
            for j in range(CurRowT):
                if CurPinT[j] == p1:
                    Xcurp[0] = CurTable[j]
                    del(CurTable[j])
                    delete(CurPinT, j)
                    CurD = Dx[j]  # 记录D(x,Mp)
                    delete(Dx, j)
                    delete(RpCur, j)
                    break
            # 如果当前水平L是最终水平,检验规则2
            if CurL == L:
                pass
            else:
                CurL += 1
                break


























