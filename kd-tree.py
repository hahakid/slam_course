import numpy as np
import open3d as o3d
from binarytree import tree, build
import time
point_number=100
point_degree=6
#print(np.floor(time.time()))
rng=np.random.RandomState(int(np.floor(time.time())))
#rng=np.random.RandomState(10)
X=rng.random_sample((point_number, point_degree)) #
X=np.floor(X*10)  # 精度
#print(X)

dataaa = np.asarray([[7., 0.],
                    [6., 7.],
                    [4., 2.],
                    [1., 7.],
                    [1., 0.],
                    [6., 9.],
                    [0., 5.],
                    [8., 6.]])


dataa = np.asarray([[5, 7, 6],
                    [5, 4, 6],
                    [4, 8, 9],
                    [3, 7, 5],
                    [5, 9, 0],
                    [0, 0, 8],#])
                    [7, 8, 9],
                    [7, 4, 7],
                    [1, 6, 1],
                    [9, 5, 4]])

data1 = np.asarray([[2, 3],
                    [5, 4],
                    [9, 6],
                    [4, 7],
                    [8, 1],
                    [7, 2]])

# utils.pc_show([data])  # show
tree = []
def findSplit(data):
    if data.shape[0] == 0:
        return -1 # None

    max_var = -9999
    colidx = 0
    # 方差最大的列
    for i in range(0, data.shape[1]):
        split = data[:, i]
        if np.var(split) > max_var:
            max_var = np.var(split)
            colidx = i

    sortdata = sorted(data, key=lambda x: x[colidx])  # 基于对应列排序
    # print(sortdata)
    # 基于方差最大列的中位数切分data
    left = []
    right = []
    med = int(np.floor(len(sortdata)/2))  # medium

    for i in range(0,len(sortdata)):
        if i < med:
            left.append(sortdata[i])
        if i == med:
            divide = sortdata[i]
        if i > med:
            right.append(sortdata[i])

    left = np.array(left)
    right = np.array(right)

    tree.append(divide)
    # 递归
    findSplit(left)
    findSplit(right)
    # return divide,left,right

def showtree(dataarray):
    for i in range(0, dataarray.shape[0]):
        print(i, ":", dataarray[i])

    findSplit(dataarray)
    treeidx=[]
    for i in tree:
        for j, d in enumerate(dataarray):
            if (d == i).all():
                treeidx.append(j)
    print(treeidx)
    root = build(treeidx)
    print(root)

#showtree(data1)
#showtree(dataa)
showtree(X)