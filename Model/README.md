# 须知

每次测试的结果将写入result.csv

## 0. 数据准备

data.csv

## 1. 预处理
```Bash
python preprocess.py
```
data_train.csv
data_test.csv

neg_chars.csv
neg_jieba.csv
pos_chars.csv
pos_jieba.csv



## 规则过滤
```Bash
python FilterRules.py
```

- 不使用规则过滤，则在`__main__`中将`filtering`函数注释，并去掉`noFiltering`的注释
- 使用探索模式（评估过滤器效果），则在`__main__`中将`filtering`函数注释，并去掉`exploring`的注释
- 如需更改过滤器的规则，则更改`toFilter`函数

## 来自语言模型的特征

### (1) 训练语言模型
```Bash
python LangModelMgr.py
```

### (2) 特征工程
```Bash
python FeatureEngr.py
```

### (3) 特征筛选

### (4) 判别式模型
```Bash
python DiscriminantModel.py
```

``/Model``



### 使用词向量的

### (1) 获得词向量
```Bash
python TrainVector.py
```
data_train_vector.vec
data_test_vector.vec

- 这里默认使用文档级的 Doc2Vec
- 文档级别的Word2Vec （尚未实现）
- 词表级别的WordList2Vec （尚未实现）

### (2) 生成式模型
```Bash
python GenerativeModel.py
```

- 默认使用SVM模型，可选LR或MLP

## 深度学习

```Bash
python DeepNet.py
```

- 默认使用fasttext






