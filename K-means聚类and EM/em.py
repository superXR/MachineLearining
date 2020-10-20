# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/10/19 21:42
@file: em.py
@desc: 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
np.random.seed(0)  # 确定随机种子，每次运行程序得到相同的训练集
P_set1 = np.random.multivariate_normal(mean1, cov1, 300)
label_A = np.array([1] * 300)  # 随机产生300个A类点,用1表示

mean2 = [1, 2]
cov2 = [[1, 0], [0, 2]]
np.random.seed(1)  # 确定随机种子，每次运行程序得到相同的训练集
P_set2 = np.random.multivariate_normal(mean2, cov2, 200)
label_B = np.array([0] * 200)  # 随机产生200个B类点,用0表示

x1, y1 = P_set1.T
x2, y2 = P_set2.T
plt.axis()
plt.title("TrainP_set")
plt.scatter(x1, y1, c='r', label='A')
plt.scatter(x2, y2, c='b', label='B')
plt.legend()
plt.show()  # 画图显示

TrainP_set = np.append(P_set1, P_set2, axis=0)
Train_label = np.append(label_A, label_B)  # 生成训练点集合和对应标签集合
def EM(dataset):
    gmm = GaussianMixture(n_components=2, random_state=12)
    gmm.fit(dataset)
    Pre_labels = gmm.predict(dataset)
    return Pre_labels


def plot_classifier(x, y, title='Classifier boundaries'):
    # 定义范围以绘制图形
    x_min, x_max = min(x[:, 0]) - 1.0, max(x[:, 0]) + 1.0
    y_min, y_max = min(x[:, 1]) - 1.0, max(x[:, 1]) + 1.0

    # 表示将在网格中使用的步长
    step_size = 0.1

    # 定义网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出
    mesh_output = EM(np.c_[x_values.ravel(), y_values.ravel()])

    # 重塑数组
    mesh_output = mesh_output.reshape(x_values.shape)

    # 使用彩色图绘制输出
    plt.figure()

    # 设置标题
    plt.title(title)

    # plt.pcolormesh()会根据模型预测的结果自动在cmap里选择颜色，画出决策面
    # cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0A0FF'])
    cm_light = mpl.colors.ListedColormap(['#A0A0FF', '#FFA0A0'])
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=cm_light)

    # 在图上叠加使用的点
    cm_dark = mpl.colors.ListedColormap(['b', 'r'])
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='black', linewidth=1, cmap=cm_dark)

    # 指定图的边界
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.show()


plot_classifier(TrainP_set, Train_label, title='EM')
