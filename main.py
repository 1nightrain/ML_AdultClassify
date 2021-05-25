import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# 读入adult数据集
with open('adult.data', 'r') as csvFile:
    readerData = pd.read_csv(csvFile)

# 分析数据集中各特征类型
# for i in readerData.columns:
#     print(f"The column {i}'s dtype is {readerData.loc[:, i].dtype}")

# 将特征分为obj类（字符串）和int类（连续变量）
objtype = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
inttype = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# 获取标签
target_train = readerData['salary']  # 从数据集中分离标签和特征
readerData.drop(['salary'], axis=1, inplace=True)  # 从原数据集中删除标签列

# 独热编码
one_hotMatrixData = readerData.copy()
one_hotMatrixData[inttype] = readerData[inttype]
one_hotMatrixData[objtype] = readerData[objtype]
one_hotMatrixData2 = pd.get_dummies(one_hotMatrixData)  # One-Hot Encode
readerData = one_hotMatrixData2

# 数据标准化
x_scale = scale(readerData)
x_scale = pd.DataFrame(x_scale)
x_scale.to_csv('./dataTrain.csv', sep=',', header=False, index=False)  # 特征输出
target_train.to_csv('./dataTarget.csv', sep=',', header=False, index=False)  # 标签输出
