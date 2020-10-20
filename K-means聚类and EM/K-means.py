# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/10/14 16:04
@file: K-means.py.py
@desc: 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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


# 计算欧氏距离
def eclud_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 初始化质心,先随机选取一个质心，再通过最远距离法选取第二个质心
def centroid(dataset):
    k_list = []  # 初始质心索引集合
    np.random.seed(1)
    k_1 = np.random.randint(0, 500)
    k_list.append(k_1)
    ''' 随机法选取k
    np.random.seed(12)
    k_2 = np.random.randint(0, 500)
    k_list.append(k_2)
    '''
    # ''' 最远距离法选取k
    dist_list = []
    dist = eclud_dist(dataset[k_1], dataset[0])
    dist_list.append(dist)
    k_list.append(0)
    for j in range(1, 500):
        dist = eclud_dist(dataset[k_1], dataset[j])
        if dist > dist_list[0]:
            dist_list[0] = dist
            k_2 = j
            k_list.append(k_2)
    # '''
    print(k_1, k_2)
    return dataset[k_list[0]], dataset[k_list[1]]


def KMeans(dataset):
    cluster_lable = []
    A_set = np.array([[0, 0]])
    B_set = np.array([[0, 0]])
    clusterChange = True
    y_1, y_2 = centroid(dataset)  # 初始化质心
    epoch = 0
    while (clusterChange):
        for i in range(dataset.shape[0]):
            if (eclud_dist(dataset[i], y_1) < eclud_dist(dataset[i], y_2)):
                A_set = np.append(A_set, [dataset[i]], axis=0)
                cluster_lable = np.append(cluster_lable, 1)
            else:
                B_set = np.append(B_set, [dataset[i]], axis=0)
                cluster_lable = np.append(cluster_lable, 0)
        A_set = np.delete(A_set, 1, axis=0)
        B_set = np.delete(B_set, 1, axis=0)
        # 更新质心
        # print(B_set)
        if all(y_1 == np.mean(A_set, axis=0)) and all(y_2 == np.mean(B_set, axis=0)):
            clusterChange = False
            epoch += 1
        else:
            y_1 = np.mean(A_set, axis=0)
            y_2 = np.mean(B_set, axis=0)
            cluster_lable = []
            A_set = np.array([[0, 0]])
            B_set = np.array([[0, 0]])
            epoch += 1
    print(epoch)
    return cluster_lable, A_set, B_set


def plot_classifier(x, y, title='Classifier boundaries'):
    # 定义范围以绘制图形
    x_min, x_max = min(x[:, 0]) - 1.0, max(x[:, 0]) + 1.0
    y_min, y_max = min(x[:, 1]) - 1.0, max(x[:, 1]) + 1.0

    # 表示将在网格中使用的步长
    step_size = 0.1

    # 定义网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出
    mesh_output, A, B = KMeans(np.c_[x_values.ravel(), y_values.ravel()])

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


Pre_labels, A, B = KMeans(TrainP_set)
Pre_sets = np.append(A, B, axis=0)
print(Pre_sets.shape, Pre_labels.shape)
plot_classifier(Pre_sets, Pre_labels, title='K-Means')
