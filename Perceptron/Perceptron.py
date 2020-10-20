# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/10/11 23:26
@file: Perceptron.py
@desc: 
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
x1 = np.array([[-6.69981438e-0, 6.60728608e-01],
               [-9.59748155e-02, 1.21837517e+00],
               [1.09221858e+00, -8.42797622e-01],
               [3.76191254e-02, 9.41110164e-01],
               [9.09293170e-01, -1.16953996e+00],
               [-1.61246018e-01, 1.04585258e+00],
               [-6.96733303e-01, 9.48857962e-01],
               [-6.10473401e-01, 1.07047058e+00],
               [-1.66657506e+00, 1.26994355e+00],
               [5.29378529e-01, 1.00556159e+00],
               [-5.39032879e-01, 1.24010824e+00]])
y1 = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1])
print(x1)
print(y1)
# 打乱顺序
np.random.seed(1)
permutation = np.random.permutation(y1.shape[0])
x2 = x1[permutation, :]
y2 = y1[permutation]
print(x2)
print(y2)

# n_samples:生成样本的数量

# n_features=2:生成样本的特征数，特征数=n_informative（） + n_redundant + n_repeated

# n_informative：多信息特征的个数

# n_redundant：冗余信息，informative特征的随机线性组合

# n_clusters_per_class ：某一个类别是由几个cluster构成的
# 训练数据和测试数据
def perceptron(x, y):
    x_data_train = x[:10, :]
    y_data_train = y[:10]

    # 正例和反例
    positive_x1 = [x[i, 0] for i in range(10) if y[i] == 1]
    positive_x2 = [x[i, 1] for i in range(10) if y[i] == 1]
    negetive_x1 = [x[i, 0] for i in range(10) if y[i] == 0]
    negetive_x2 = [x[i, 1] for i in range(10) if y[i] == 0]

    # 定义感知机
    clf = Perceptron(fit_intercept=False, shuffle=False)
    # 使用训练数据进行训练
    clf.fit(x_data_train, y_data_train)
    # 得到训练结果，权重矩阵
    print(clf.coef_)
    print(clf.n_iter_)

    # 画出正例和反例的散点图
    plt.scatter(positive_x1, positive_x2, c='red')
    plt.scatter(negetive_x1, negetive_x2, c='blue')
    # 画出超平面（在本例中即是一条直线）
    line_x = np.arange(-4, 4)
    line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
    plt.plot(line_x, line_y)
    plt.show()


print("打乱顺序前：")
perceptron(x1, y1)
print("打乱顺序后：")
perceptron(x2, y2)
