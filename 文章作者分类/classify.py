from jieba.analyse import *

A_list = []
B_list = []  # 分别存放A和B的关键词
with open('A_article.txt') as f:
    data = f.read()
    for A_keyword, A_weight in extract_tags(data, topK=100, withWeight=True):
        A_list.append(A_keyword)  # A类关键词
        # print('%s %s' % (A_keyword,A_weight))   #打印出A类文章权重前40的关键词和权重
with open('B_article.txt') as f:
    data = f.read()
    for B_keyword, B_weight in extract_tags(data, topK=100, withWeight=True):
        B_list.append(B_keyword)  # B类关键词
        # print('%s %s' % (B_keyword, B_weight))  #打印出A类文章权重前40的关键词和权重


def classify(x):
    with open(str(x) + '_article.txt') as f:
        data = f.read()
        a = 0;
        b = 0;
        for keyword, weight in extract_tags(data, topK=100, withWeight=True):
            if keyword in A_list:
                a += 1
            else:
                b += 1
        if a > b:
            print('第' + str(x) + '文章的作者是汉密尔顿')
        else:
            print('第' + str(x) + '文章的作者是麦迪逊')


for x in range(1, 12):
    classify(x)
