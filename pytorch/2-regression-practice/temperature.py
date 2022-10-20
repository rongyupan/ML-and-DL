# author : 'wangzhong';
# date: 04/01/2021 15:56
"""
pytorch基础气温预测项目
"""

from audioop import bias
import numpy as np
# conda install pandas
import pandas as pd
import matplotlib.pyplot as plt
import torch
# conda install scikit-learn
from sklearn import preprocessing
import datetime

# 通过python console查看维度
features = pd.read_csv('temps.csv')

# 看看数据长什么样子
print(features.head())

# week 列是字符串，不方便处理，所以做独特编码
features = pd.get_dummies(features)
print(features.head())

# 标签
labels = np.array(features['actual'])

# 在特征中去掉标签
features = features.drop('actual', axis=1)

# 名字单独保存一下，以备后患
feature_list = list(features.columns)
# 特征标准化处理：维度[348,14]
input_features = preprocessing.StandardScaler().fit_transform(features)

# 转为tensor格式
x = torch.tensor(input_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)

learning_rate = 0.001
losses = []
predict = []

# 手写梯度下降
handwrite = 0
if(handwrite == 1):
  # 权重参数初始化，一层hidden layer
  # res_1: relu([348,14]*[14,128]+bias)
  # res_2: res_1*[128,1]+(1)
  weights = torch.randn((14, 128), dtype=torch.float, requires_grad=True)
  biases = torch.randn(128, dtype=torch.float, requires_grad=True)
  weights2 = torch.randn((128, 1), dtype=torch.float, requires_grad=True)
  biases2 = torch.randn(1, dtype=torch.float, requires_grad=True)

  # 简单的手写梯度下降
  for i in range(1000):
      # 计算隐层，mm为矩阵乘法。[348,14]*[14,128]
      # biases 会加到 [348,128] 中的每个元素上
      hidden = x.mm(weights) + biases
      # 加入激活函数
      hidden = torch.relu(hidden)
      print(hidden.shape)
      # 预测结果。[348,128]*[128,1]
      predictions = hidden.mm(weights2) + biases2
      # 代价函数为均方误差
      loss = torch.mean((predictions - y) ** 2)
      losses.append(loss.data.numpy())

      # 打印损失值
      if i % 100 == 0:
          print('loss:', loss)
      # 返向传播计算
      loss.backward()

      # 更新参数：反向梯度，快速下降法
      weights.data.add_(-learning_rate*weights.grad.data)
      biases.data.add_(-learning_rate*biases.grad.data)
      weights2.data.add_(-learning_rate*weights2.grad.data)
      biases2.data.add_(-learning_rate*biases2.grad.data)

      # 进行下一次迭代前，清空当前梯度，防止累加
      weights.grad.data.zero_()
      biases.grad.data.zero_()
      weights2.grad.data.zero_()
      biases2.grad.data.zero_()
else:
    # 更简单的，全部用 torch 内置函数完成的模型训练
    # input_features=[348,14]，取出14
    input_size = input_features.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 16
    # Linear 需要指定输入输出的列数
    # [348,14] → my_nn → [348,1]
    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size),
    )
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=learning_rate)

    for i in range(1000):
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(input_features), batch_size):
            end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
            # xx: [16,14]
            xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
            # yy: [16] 这是个数组，scalar 类型
            yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
            
            prediction = my_nn(xx)
            # 降为一维，否则loss会告警
            # 从 [16,1] → [16]，和 yy 契合
            prediction = prediction.squeeze(-1)
            # 计算损失值
            loss = cost(prediction, yy)
            # 初始化梯度
            optimizer.zero_grad()
            # 后向传播
            loss.backward(retain_graph=True)
            # 更新权值
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        # 打印损失
        if i % 100 == 0:
            losses.append(np.mean(batch_loss))
            print(i, np.mean(batch_loss))

    predict = my_nn(x).data.numpy()


# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']
# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
# 使用 pandas 先把数据保存到 DataFrame 中，再直接调用画图就比较方便
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
predictions_data = pd.DataFrame(data={'date': dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()

# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
