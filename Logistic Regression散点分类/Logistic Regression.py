# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/9/24 9:40
@file: Logistic Regression.py
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt

mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
P_set1 = np.random.multivariate_normal(mean1, cov1, 300)
label_A = np.array([1] * 300)  # 随机产生300个A类点,用1表示

mean2 = [1, 2]
cov2 = [[1, 0], [0, 2]]
P_set2 = np.random.multivariate_normal(mean2, cov2, 200)
label_B = np.array([0] * 200)  # 随机产生200个B类点,用0表示

x1, y1 = P_set1.T
x2, y2 = P_set2.T
plt.axis()
plt.title("TrainP_set")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x1, y1, label='A')
plt.scatter(x2, y2, c='r', label='B')
plt.legend()
plt.show()  # 画图显示

TrainP_set = np.append(P_set1, P_set2, axis=0)
Train_label = np.append(label_A, label_B)  # 生成训练点集合和对应标签集合

TestP_set = np.append(np.random.multivariate_normal(mean1, cov1, 300), np.random.multivariate_normal(mean2, cov2, 200),
                      axis=0)
Test_label = Train_label  # 生成测试点集合和对应标签
'''
print(TrainP_set.shape)
print(Train_label.shape)
'''


# 定义模型函数
def model(theta, x):
    z = x.dot(theta)
    return 1.0 / (1 + np.exp(-z))


# 定义损失函数
def cost_function(theta, x, y):
    m = y.size
    h = model(theta, x)
    j = -(1.0 / m) * ((np.log(h)).T.dot(y) + (np.log(1 - h)).T.dot(1 - y))
    return j[0, 0]


# GD方法梯度下降,更新权重𝜃
def update_GD(theta, x, y, Learning_rate):
    m = y.size
    h = model(theta, x)
    grad = (1.0 / m) * x.T.dot(h - y)
    return theta - Learning_rate * grad


# SGD方法梯度下降,更新权重𝜃
def update_SGD(theta, x, y, Learning_rate):
    # m = y.size
    # x = np.array([x[i]])
    h = model(theta, x)
    grad = x.T.dot(h - y)
    return theta - Learning_rate * grad


# 定义逻辑回归算法
def logistic_regression(x, y, epoch, g):
    m = x.shape[0]  # 样本个数
    n = x.shape[1]  # 样本特征的个数
    y = y.reshape(m, 1)
    cost_list = []  # 记录代价函数的值
    Learning_rate = 0.01  # 学习率

    i = 0
    if g == 'GD':
        theta = np.zeros(n).reshape(n, 1)  # 设置权重参数的初始值
        cost = cost_function(theta, x, y)  # 得到损失函数的初始值
        cost_list.append(cost)
        while (i < epoch):
            theta = update_GD(theta, x, y, Learning_rate)  # 更新权值
            cost = cost_function(theta, x, y)
            cost_list.append(cost)
            i += 1
        return theta, cost_list

    elif g == 'SGD':
        theta = np.zeros(n).reshape(n, 1)  # 设置权重参数的初始值
        # for j in range(500):
        while (i < epoch):
            j = np.random.randint(0, 500)
            each_x = np.array([x[j]])
            each_y = np.array([y[j]])
            if i == 0:
                cost = cost_function(theta, each_x, each_y)  # 得到损失函数的初始值
                cost_list.append(cost)
            theta = update_SGD(theta, each_x, each_y, Learning_rate)  # 更新权值
            cost = cost_function(theta, each_x, each_y)
            cost_list.append(cost)
            i += 1
    return theta, cost_list


# 定义预测函数
def predict(theta, x, threshold=0.5):
    Pr_A = np.array([[0, 0]])
    Pr_B = np.array([[0, 0]])
    p = model(theta, x)
    for i in range(500):
        if p[i] >= threshold:
            Pr_A = np.append(Pr_A, [x[i]], axis=0)
        else:
            Pr_B = np.append(Pr_B, [x[i]], axis=0)
    x1, y1 = Pr_A.T
    x2, y2 = Pr_B.T
    plt.axis()
    plt.title("PredictP_set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x1, y1, label='A')
    plt.scatter(x2, y2, c='r', label='B')
    plt.legend()
    plt.show()  # 画图显示
    return Pr_A, Pr_B


def main(epoch):
    theta, cost_list = logistic_regression(TrainP_set, Train_label, epoch, g='GD')
    predict(theta, TrainP_set)
    # print('GD训练集的权重:', theta)
    # print(cost_list)

    theta, cost_list = predict(theta, TestP_set)
    # print('GD测试集的权重:', theta)
    # print(cost_list)

    theta, cost_list = logistic_regression(TrainP_set, Train_label, epoch, g='SGD')
    predict(theta, TrainP_set)
    # print('SGD训练集的权重:', theta)
    # print(cost_list)

    theta, cost_list = predict(theta, TestP_set)
    # print('SGD测试集的权重:', theta)
    # print(cost_list)


if __name__ == '__main__':
    main(100)
    main(300)
    main(500)
    main(800)
    main(1000)
