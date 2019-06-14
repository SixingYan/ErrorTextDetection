# 须知

每次测试的结果将写入result.csv

## 0. 数据准备
至少包含 [sent, target]
data.csv

## 1. 预处理
```Bash
python preprocess.py
```
生成[sent,sent_chars,sent_words,target]

data_train.csv
data_test.csv


## 2. 规则过滤
```Bash
python FilterRules.py
```
（这将会增加一列``['isFilter']``，默认为``None``若被过滤则显示违反的规则，如``_islen``）

- 使用探索模式（评估过滤器效果），则在`__main__`中将`filtering`函数注释，并去掉`exploring`的注释
- 如需更改过滤器的规则，则更改`toFilter`函数

```Bash
python FilterRules.py -task exploring
```
探索模式将会评估当前规则的准确率

## 来自语言模型的特征

### (1) 训练语言模型
```Bash
python LangModelMgr.py
```

```Bash
python LangModelMgr.py -n 2 -dtype words -dsource std -dname weibo
```

### (2) 特征工程
```Bash
python FeatureEngr.py
```
data_train_feat.csv
data_test_feat.csv


### (3) 特征筛选
```Bash
python Visualization.py
```
生成关于特征和标签之间的 皮尔森相关系数热力图

```Bash
python Visualization.py -plot len l3_neg_ppl
```

```Bash
python FeatureEngr.py -del len 
```



### (4) 判别式模型
```Bash
python DiscriminantModel.py
```

``/Model``



## 基于词向量

### (1) 获得词向量
```Bash
python ToVectorMgr.py
```

data_train_d2v.vec
data_test_d2v.vec

- 这里默认使用文档级的 Doc2Vec
- 文档级别的Word2Vec （尚未实现）
- 词表级别的WordList2Vec （尚未实现）

### (2) 生成式模型
```Bash
python GenerativeModel.py
```

- 默认使用SVM模型，可选LR或MLP



## 神经网络

```Bash
python DeepNet.py
```

- 默认使用fasttext


```Bash
python DeepNet.py -net textcnn
```



## 集成学习

```
python Ensenmble.py
```

