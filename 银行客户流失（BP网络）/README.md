# 银行客户流失预测

## 基本思路

这是一个用于客户流失预测的神经网络模型，其主要步骤包括数据准备、模型构建、训练和评估。

1. **数据准备**：
   - 从CSV文件中读取数据，数据包括客户的一些特征信息（如信用分数、年龄、存贷款情况等）以及是否已流失（标签数据）。
   - 将读取的特征数据进行向量化，同时对标签数据进行One-Hot编码。

2. **模型构建**：
   - 使用TensorFlow构建一个简单的***前馈神经网络***（Feedforward Neural Network），包括三个隐藏层和一个输出层。
   - 使用Sigmoid作为激活函数，在每个隐藏层之后进行Dropout操作以防止过拟合。
   - 输出层使用Softmax激活函数，用于多分类任务。

3. **训练模型**：
   - 定义损失函数为交叉熵损失（softmax_cross_entropy_with_logits_v2）。
   - 使用Adam优化器进行优化。
   - 使用Mini-batch训练，通过迭代多次训练，不断调整模型参数以减小损失。
   - 训练过程中，记录训练集和测试集的准确率和损失值，并定期输出以监控模型训练的进度。

4. **评估模型**：
   - 在每个训练周期结束时，评估模型在训练集和测试集上的准确率。
   - 使用Matplotlib将训练过程中的准确率和损失值绘制成曲线，以便可视化分析模型的性能。

5. **保存模型**（注释中的代码）：
   - 在训练过程中，可以保存训练好的模型，以便后续使用。

这个模型通过对客户的一些特征信息进行训练，来预测客户是否会流失，以帮助银行等机构做出相关的决策。

### 数据预处理

重点在于数据预处理
- 归一化处理：将属性转化为0-1之间的数值

在这段代码中，数据预处理的过程包括以下步骤：

1. **读取数据**：
   从两个 CSV 文件中读取客户数据和测试数据。这些数据包括客户的一些特征信息，如信用分数、年龄、存贷款情况等，以及是否已流失（标签数据）。

2. **构建特征向量**：
   首先，根据代码中的循环，将客户数据中的特征值取出，并构建为一个列表。例如：
   ```python
   mid = []
   mid.append(df["Geography"][i])
   mid.append(df["Gender"][i])
   mid.append(df["EB"][i])
   mid.append(df["Age"][i])
   mid.append(df["EstimatedSalary"][i])
   mid.append(df["NumOfProducts"][i])
   mid.append(df["CreditScore"][i])
   mid.append(df["Tenure"][i])
   mid.append(df["HasCrCard"][i])
   mid.append(df["IsActiveMember"][i])
   ```

3. **标签处理**：
   对于标签数据（即是否已流失），进行 One-Hot 编码。使用 `OneHotEncoder` 对标签数据进行编码。

4. **数据洗牌**：
   使用 `shuffle()` 函数对训练数据和标签进行洗牌操作，以增加模型训练的随机性和稳健性。

5. **数据转换**：
   将构建好的特征列表和标签列表转换为 NumPy 数组，以便后续在 TensorFlow 中进行处理。

6. **占位符定义**：
   在 TensorFlow 中定义了三个占位符，分别用于接收特征数据、标签数据和 Dropout 操作的保留概率。

总的来说，数据预处理的目的是将原始数据转换为神经网络模型可以接受的格式，并进行必要的编码和转换以适应模型的训练和评估过程。

### 模型构建（神经网络的搭建过程）

让我们逐步解释代码中神经网络的搭建过程：

```python
# 定义输入占位符
x = tf.placeholder(tf.float32, shape=(None, 10))
# # 二分类问题 [0,1]
y = tf.placeholder(tf.float32, shape=(None, 2))
keep = tf.placeholder(tf.float32)
```
这里首先定义了三个占位符 `x`、`y` 和 `keep`，分别用于接收输入特征、标签以及用于控制 Dropout 操作的保留概率。

```python
# layer1
var1 = tf.Variable(tf.truncated_normal([10, 256], stddev=0.1))
bias1 = tf.Variable(tf.zeros([256]))
hc1 = tf.add(tf.matmul(x, var1), bias1)
h1 = tf.sigmoid(hc1)
h1 = tf.nn.dropout(h1, keep_prob=keep)
```
接下来，定义了第一个隐藏层，包括权重矩阵 `var1` 和偏置向量 `bias1`。使用 `tf.truncated_normal` 初始化权重矩阵，采用截断正态分布，标准差为 0.1。然后，通过矩阵乘法 `tf.matmul` 将输入 `x` 与权重矩阵相乘，再加上偏置向量 `bias1` 得到隐藏层的加权输入 `hc1`。接着，使用 Sigmoid 激活函数 `tf.sigmoid` 对加权输入进行非线性变换得到隐藏层的输出 `h1`。最后，使用 `tf.nn.dropout` 进行 Dropout 操作，以减少过拟合。

```python
# layer2
var2 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.1))
bias2 = tf.Variable(tf.zeros([256]))
hc2 = tf.add(tf.matmul(h1, var2), bias2)
h2 = tf.sigmoid(hc2)
h2 = tf.nn.dropout(h2, keep_prob=keep)
```
类似地，定义了第二个隐藏层，包括权重矩阵 `var2` 和偏置向量 `bias2`。在该隐藏层中，输入为第一个隐藏层的输出 `h1`。通过矩阵乘法和激活函数操作得到第二个隐藏层的输出 `h2`。

```python
# layer3
var3 = tf.Variable(tf.truncated_normal([256, 2], stddev=0.1))
bias3 = tf.Variable(tf.zeros([2]))
hc3 = tf.add(tf.matmul(h2, var3), bias3)
h3 = tf.nn.softmax(hc3)
```
最后，定义了输出层，包括权重矩阵 `var3` 和偏置向量 `bias3`。输出层的激活函数使用了 Softmax 函数 `tf.nn.softmax`，将神经网络的输出转换为各个类别的概率分布。

总结一下，这段代码构建了一个拥有两个隐藏层的前馈神经网络。每个隐藏层都采用 Sigmoid 激活函数，并通过 Dropout 操作来防止过拟合。最后一个输出层使用 Softmax 函数，用于多分类问题。

### 模型训练

训练过程主要涉及以下步骤：

1. **初始化变量和优化器**：
   在 TensorFlow 会话中初始化变量，包括神经网络中的权重和偏置，以及优化器（这里使用的是 Adam 优化器）。

2. **Mini-batch 训练**：
   通过迭代多次训练，不断调整模型参数以减小损失。每次迭代都会从训练集中随机选择一小部分数据（Mini-batch）（小批量梯度下降法），将其输入到神经网络中进行前向传播和反向传播。通过计算损失函数并调用优化器的 `minimize` 方法来更新网络中的权重和偏置，从而最小化损失。
   小批量梯度下降：每次迭代都会从训练集中随机选择一小部分数据（Mini-batch），将其输入到神经网络中进行前向传播和反向传播。通过计算损失函数并调用优化器的 `minimize` 方法来更新网络中的权重和偏置，从而最小化损失。

3. **评估模型性能**：
   在每个训练周期结束时，评估模型在训练集和测试集上的准确率。通过调用 `sess.run()` 方法，传入 `acc` 占位符和相应的输入数据，计算模型在当前数据上的准确率。

4. **可视化训练过程**（可选）：
   在训练过程中，还可以记录训练集和测试集的准确率和损失值，并定期输出以监控模型训练的进度。这里使用 Matplotlib 库将训练过程中的准确率和损失值绘制成曲线，以便直观地分析模型的性能。

5. **保存模型**（可选）：
   在训练过程中，还可以选择保存训练好的模型，以便后续使用。这里使用 TensorFlow 提供的 `tf.train.Saver` 类来保存模型参数。

总的来说，训练过程通过多次迭代，不断调整神经网络中的参数，使其逐渐收敛到最优解，从而实现对训练数据的拟合，并最终达到较好的泛化能力。