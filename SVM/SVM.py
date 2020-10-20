# encoding: utf-8
"""
@author: 徐瑞
@time: 2020/9/29 23:17
@file: SVM.py
@desc: 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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


def train(params, data):
    classifier = SVC(**params)  # 使用选择的核函数构建分类器
    classifier.fit(TrainP_set, Train_label)  # 训练模型
    if data == 'Train':
        print(classifier.score(TrainP_set, Train_label))  # 打印训练集精确率
    else:
        print(classifier.score(TestP_set, Test_label))  # 打印测试集精确率
    return classifier


# 绘制分类的边界和结果
def plot_classifier(classifier, x, y, title='Classifier boundaries'):
    # 定义范围以绘制图形
    x_min, x_max = min(x[:, 0]) - 1.0, max(x[:, 0]) + 1.0
    y_min, y_max = min(x[:, 1]) - 1.0, max(x[:, 1]) + 1.0

    # 表示将在网格中使用的步长
    step_size = 0.01

    # 定义网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    print(np.c_[x_values.ravel(), y_values.ravel()].shape)

    # 重塑数组
    mesh_output = mesh_output.reshape(x_values.shape)

    # 使用彩色图绘制输出
    plt.figure()

    # 设置标题
    plt.title(title)

    # plt.pcolormesh()会根据模型预测的结果自动在cmap里选择颜色，画出决策面
    cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0A0FF'])
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=cm_light)

    # 在图上叠加使用的点
    cm_dark = mpl.colors.ListedColormap(['r', 'b'])
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='black', linewidth=1, cmap=cm_dark)

    # 指定图的边界
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.show()


'''
params = {'kernel': 'linear'}  # 构建线性分类器
params = {'kernel': 'poly', 'degree': 3}  # 构建非线性分类器，核函数为3次多项式
params = {'kernel': 'rbf'}  # 用rbf高斯核函数建立非线性分类器
'''
plot_classifier(train(params={'kernel': 'linear'}, data='Train'), TrainP_set, Train_label,
                title='Training data classification')
plot_classifier(train(params={'kernel': 'linear'}, data='Test'), TestP_set, Test_label,
                title='Test data classification')

plot_classifier(train(params={'kernel': 'poly', 'degree': 3}, data='Train'), TrainP_set, Train_label,
                title='Training data classification')
plot_classifier(train(params={'kernel': 'poly', 'degree': 3}, data='Test'), TestP_set, Test_label,
                title='Test data classification')

plot_classifier(train(params={'kernel': 'rbf'}, data='Train'), TrainP_set, Train_label,
                title='Training data classification')
plot_classifier(train(params={'kernel': 'rbf'}, data='Test'), TestP_set, Test_label, title='Test data classification')
