#!/usr/bin/env python
# coding: utf-8
# 18崇新 徐瑞 201800121045
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
P_set1 = np.random.multivariate_normal(mean1, cov1, 300)
label_A = np.array(['A'] * 300)

mean2 = [1, 2]
cov2 = [[1, 0], [0, 2]]
P_set2 = np.random.multivariate_normal(mean2, cov2, 200)
label_B = np.array(['B'] * 200)

x1, y1 = P_set1.T
x2, y2 = P_set2.T
plt.axis()
plt.title("TrainP_set")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x1, y1, label='A')
plt.scatter(x2, y2, c='r', label='B')
plt.legend()
plt.show()

TrainP_set = np.append(P_set1, P_set2, axis=0)
Train_label = np.append(label_A, label_B)
# 重新随机生成一组满足题意的随机点作为测试集
TestP_set = np.append(np.random.multivariate_normal(mean1, cov1, 300), np.random.multivariate_normal(mean2, cov2, 200),
                      axis=0)
Test_label = Train_label
'''
print(TrainP_set.shape)
print(Train_lable.shape)
'''


# 下面进行KNN分类
def KNN(k, TrainP_set, TestP_set):
    Sort_label = []  # 分类后的标签初始化
    for i in range(500):
        x, y = TestP_set[i]
        length = []
        np.array(length)
        for j in range(500):
            m = (x - TrainP_set[j][0]) ** 2
            n = (y - TrainP_set[j][1]) ** 2
            length = np.append(length, np.sqrt(m + n))  # 距离采用欧式距离
        index = np.argsort(length)  # 获取排序后的索引
        top_lable = [Train_label[i] for i in index[:k]]  # 获取前k个最近邻点的标签
        predict_label = Counter(top_lable).most_common(1)[0][0]  # 投票，找出出现最多的点的标签
        Sort_label = np.append(Sort_label, predict_label)
    # print(Sort_label)
    Pd_A = np.array([[0, 0]])  # 分类后的A点集合
    Pd_B = np.array([[0, 0]])  # 分类后的B点集合
    for i in range(500):
        if Sort_label[i] == 'A':
            Pd_A = np.append(Pd_A, [TestP_set[i]], axis=0)
        elif Sort_label[i] == 'B':
            Pd_B = np.append(Pd_B, [TestP_set[i]], axis=0)
    # print(Pd_A)
    # print(Pd_B)
    # print(Sort_label)
    x1, y1 = Pd_A.T
    x2, y2 = Pd_B.T
    plt.axis()
    plt.title("Predicted(k=" + str(k) + ')')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x1, y1, label='A')
    plt.scatter(x2, y2, c='r', label='B')
    plt.legend()
    plt.show()
    # 计算分类准确率
    error = 0
    for i in range(500):
        if Sort_label[i] != 'A' and i < 300:
            error += 1
        elif Sort_label[i] != 'B' and i > 300:
            error += 1
    correct_rate = 1 - (error / 500)
    print()
    print('k=' + str(k) + '时', '分类正确率为：', correct_rate)


KNN(20, TrainP_set, TestP_set)
KNN(15, TrainP_set, TestP_set)
KNN(10, TrainP_set, TestP_set)
KNN(5, TrainP_set, TestP_set)
KNN(1, TrainP_set, TestP_set)
