import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 读入标签
with open('dataTarget.csv', 'r') as csvFile:
    target = pd.read_csv(csvFile)

# 读入特征集
with open('dataTrain.csv', 'r') as csvFile:
    feature = pd.read_csv(csvFile)

score_lt = []
x = feature
y = np.ravel(target)

# 交叉验证
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i + 1, random_state=90)  # 随机森林 n_estimators：森林中树的数目
    print('%d 个分类器数量的随机森林正在建立' % i + 1)
    score = cross_val_score(rfc, x, y, cv=10).mean()  # 计算得分
    score_lt.append(score)

# 输出最大分数
score_max = max(score_lt)
print('最大得分：{}'.format(score_max), '子树数量为：{}'.format(score_lt.index(score_max) + 1))

# 绘制学习曲线
x = np.arange(0, 200, 10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.savefig('./out.jpg')
plt.show()
