# 基于预训练模型 ERNIE-Gram 实现语义匹配

本案例介绍 NLP 最基本的任务类型之一 —— 文本语义匹配，并且基于 PaddleNLP 使用百度开源的预训练模型 ERNIE1.0 为基础训练效果优异的语义匹配模型，来判断 2 个文本语义是否相同。

# 一、 背景介绍
文本语义匹配任务，简单来说就是给定两段文本的相，让模型来判断两段文本是不是语义相似。

在本案例中以权威的语义匹配数据集 [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 为例，[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 数据集是基于百度知道相似问题推荐构造的通问句语义匹配数据集。训练集中的每两段文本都会被标记为 1（语义相似） 或者 0（语义不相似）

例如百度知道场景下，用户搜索一个问题，模型会计算这个问题与候选问题是否语义相似，语义匹配模型会找出与问题语义相似的候选问题返回给用户，避免用户重复提问。例如，当某用户在搜索引擎中搜索 “深度学习的教材有哪些？”，模型就自动找到了一些语义相似的问题展现给用户:
![](https://ai-studio-static-online.cdn.bcebos.com/ecc1244685ec4476b869ce8a32d421c0ad530666e98d487da21fa4f61670544f)

# 二、快速实践

介绍如何准备数据，基于 ERNIE-Gram 模型进行匹配网络大家，然后快速进行语义匹配模型的训练、评估和预测。

## 1.数据加载
为了训练匹配模型，一般需要准备三个数据集：训练集 train.tsv、验证集dev.tsv、测试集test.tsv。此案例我们使用 PaddleNLP 内置的语义数据集 [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 来进行训练、评估、预测。

训练集: 用来训练模型参数的数据集，模型直接根据训练集来调整自身参数以获得更好的分类效果。

验证集: 用于在训练过程中检验模型的状态，收敛情况。验证集通常用于调整超参数，根据几组模型验证集上的表现决定哪组超参数拥有最好的性能。

测试集: 用来计算模型的各项评估指标，验证模型泛化能力。

## 2.LCQMC	A Large-scale Chinese Question Matching Corpus 语义匹配数据集
[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 数据集是公开的语义匹配权威数据集。PaddleNLP 已经内置该数据集，一键即可加载。

## 3.包引入


```python
# 正式开始实验之前首先通过如下命令安装最新版本的 paddlenlp
!python -m pip install --upgrade paddlenlp==2.0.2 -i https://pypi.org/simple
```

    Collecting paddlenlp==2.0.2
    [?25l  Downloading https://files.pythonhosted.org/packages/b1/e9/128dfc1371db3fc2fa883d8ef27ab6b21e3876e76750a43f58cf3c24e707/paddlenlp-2.0.2-py3-none-any.whl (426kB)
    [K     |████████████████████████████████| 430kB 25kB/s eta 0:00:012
    [?25hRequirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (2.1.1)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (4.1.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (2.9.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.4.4)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.42.1)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (1.2.2)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (3.14.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (3.8.2)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.20.3)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (2.22.0)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.15.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.1.1)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.21.0)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (7.1.2)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (0.8.53)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.0.0)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp==2.0.2) (0.3.3)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp==2.0.2) (0.24.2)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (2.6.0)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (2.2.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.6.1)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.23)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (1.25.6)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (7.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (2.10.1)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (1.1.0)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (5.1.2)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (2.0.1)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.3.0)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.3.4)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.4.10)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (16.7.9)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp==2.0.2) (3.9.9)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp==2.0.2) (0.18.0)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (2019.3)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (2.1.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (1.6.3)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (0.14.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.6.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl->paddlenlp==2.0.2) (1.1.1)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.2



```python
import time
import os
import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp

# 一键加载 Lcqmc 的训练集、验证集
train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
```

## 4.数据下载


```python
# paddlenlp 会自动下载 lcqmc 数据集解压到 "${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/" 目录下
! ls ${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc
print(paddlenlp.__version__)
```

## 5.数据查看


```python
# 输出训练集的前 20 条样本
for idx, example in enumerate(train_ds):
    if idx <= 20:
        print(example)
```

##  6.数据预处理
通过 paddlenlp 加载进来的 [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 数据集是原始的明文数据集，这部分我们来实现组 batch、tokenize 等预处理逻辑，将原始明文数据转换成网络训练的输入数据 

### 6.1定义样本转换函数


```python
# 因为是基于预训练模型 ERNIE-Gram 来进行，所以需要首先加载 ERNIE-Gram 的 tokenizer，
# 后续样本转换函数基于 tokenizer 对文本进行切分

tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
```


```python
# 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
```


```python
### 对训练集的第 1 条数据进行转换
input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
```


```python
print(input_ids)
```


```python
print(token_type_ids)
```


```python
print(label)
```


```python
# 为了后续方便使用，我们给 convert_example 赋予一些默认参数
from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512)
```

### 6.2 组装 Batch 数据 & Padding

上一小节，我们完成了对单条样本的转换，本节我们需要将样本组合成 Batch 数据，对于不等长的数据还需要进行 Padding 操作，便于 GPU 训练。

PaddleNLP 提供了许多关于 NLP 任务中构建有效的数据 pipeline 的常用 API

| API                             | 简介                                       |
| ------------------------------- | :----------------------------------------- |
| `paddlenlp.data.Stack`          | 堆叠N个具有相同shape的输入数据来构建一个batch |
| `paddlenlp.data.Pad`            | 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度 |
| `paddlenlp.data.Tuple`          | 将多个batchify函数包装在一起 |

更多数据处理操作详见： https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/docs/data.md


```python
from paddlenlp.data import Stack, Pad, Tuple
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
print("Stacked Data: \n", result)
print()

a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
print("Padded Data: \n", result)
print()

data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, labels = batchify_fn(data)
print("ids: \n", ids)
print()
print("labels: \n", labels)
print()
```


```python
# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
```

### 6.3定义 Dataloader
下面我们基于组 batchify_fn 函数和样本转换函数 trans_func 来构造训练集的 DataLoader, 支持多卡训练



```python

# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

## 7.模型搭建

自从 2018 年 10 月以来，NLP 个领域的任务都通过 Pretrain + Finetune 的模式相比传统 DNN 方法在效果上取得了显著的提升，本节我们以百度开源的预训练模型 ERNIE-Gram 为基础模型，在此之上构建 Point-wise 语义匹配网络。

首先我们来定义网络结构:


```python
import paddle.nn as nn

# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
#pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')


class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)

        return probs

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)
```

## 8. 模型训练 & 评估


```python
from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 3
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()
```


```python
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
```


```python
# 接下来，开始正式训练模型

global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):

        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        
        # 每间隔 10 step 输出训练指标
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        # 每间隔 100 step 在验证集和测试集上进行评估
        if global_step % 100 == 0:
            evaluate(model, criterion, metric, dev_data_loader, "dev")
            
# 训练结束后，存储模型参数
save_dir = os.path.join("checkpoint", "model_%d" % global_step)
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state.pdparams')
paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)
```

模型训练过程中会输出如下日志:
```
global step 5310, epoch: 3, batch: 1578, loss: 0.31671, accu: 0.95000, speed: 0.63 step/s
global step 5320, epoch: 3, batch: 1588, loss: 0.36240, accu: 0.94063, speed: 6.98 step/s
global step 5330, epoch: 3, batch: 1598, loss: 0.41451, accu: 0.93854, speed: 7.40 step/s
global step 5340, epoch: 3, batch: 1608, loss: 0.31327, accu: 0.94063, speed: 7.01 step/s
global step 5350, epoch: 3, batch: 1618, loss: 0.40664, accu: 0.93563, speed: 7.83 step/s
global step 5360, epoch: 3, batch: 1628, loss: 0.33064, accu: 0.93958, speed: 7.34 step/s
global step 5370, epoch: 3, batch: 1638, loss: 0.38411, accu: 0.93795, speed: 7.72 step/s
global step 5380, epoch: 3, batch: 1648, loss: 0.35376, accu: 0.93906, speed: 7.92 step/s
global step 5390, epoch: 3, batch: 1658, loss: 0.39706, accu: 0.93924, speed: 7.47 step/s
global step 5400, epoch: 3, batch: 1668, loss: 0.41198, accu: 0.93781, speed: 7.41 step/s
eval dev loss: 0.4177, accu: 0.89082
global step 5410, epoch: 3, batch: 1678, loss: 0.34453, accu: 0.93125, speed: 0.63 step/s
global step 5420, epoch: 3, batch: 1688, loss: 0.34569, accu: 0.93906, speed: 7.75 step/s
global step 5430, epoch: 3, batch: 1698, loss: 0.39160, accu: 0.92917, speed: 7.54 step/s
global step 5440, epoch: 3, batch: 1708, loss: 0.46002, accu: 0.93125, speed: 7.05 step/s
global step 5450, epoch: 3, batch: 1718, loss: 0.32302, accu: 0.93188, speed: 7.14 step/s
global step 5460, epoch: 3, batch: 1728, loss: 0.40802, accu: 0.93281, speed: 7.22 step/s
global step 5470, epoch: 3, batch: 1738, loss: 0.34607, accu: 0.93348, speed: 7.44 step/s
global step 5480, epoch: 3, batch: 1748, loss: 0.34709, accu: 0.93398, speed: 7.38 step/s
global step 5490, epoch: 3, batch: 1758, loss: 0.31814, accu: 0.93437, speed: 7.39 step/s
global step 5500, epoch: 3, batch: 1768, loss: 0.42689, accu: 0.93125, speed: 7.74 step/s
eval dev loss: 0.41789, accu: 0.88968
```

基于默认参数配置进行单卡训练大概要持续 4 个小时左右，会训练完成 3 个 Epoch, 模型最终的收敛指标结果如下:


| 数据集 | Accuracy |
| -------- | -------- |
| dev.tsv     | 89.62  |

可以看到: 我们基于 PaddleNLP ，利用 ERNIE-Gram 预训练模型使用非常简洁的代码，就在权威语义匹配数据集上取得了很不错的效果.

## 9.模型预测

接下来我们使用已经训练好的语义匹配模型对一些预测数据进行预测。待预测数据为每行都是文本对的 tsv 文件，我们使用 Lcqmc 数据集的测试集作为我们的预测数据，进行预测并提交预测结果到 [千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)

下载我们已经训练好的语义匹配模型, 并解压


```python
# 下载我们基于 Lcqmc 事先训练好的语义匹配模型并解压
! wget https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
! tar -xvf ernie_gram_zh_pointwise_matching_model.tar
```


```python
# 测试数据由 2 列文本构成 tab 分隔
# Lcqmc 默认下载到如下路径
! head -n10 "${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/test.tsv"
```

### 9.1定义预测函数


```python

def predict(model, data_loader):
    
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs
```

### 9.2 定义预测数据的 data_loader


```python
# 预测数据的转换函数
# predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512,
    is_test=True)

# 预测数据的组 batch 操作
# predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
): [data for data in fn(samples)]

# 加载预测数据
test_ds = load_dataset("lcqmc", splits=["test"])
```


```python
batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)

# 生成预测数据 data_loader
predict_data_loader =paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

### 9.3 定义预测模型


```python
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

model = PointwiseMatching(pretrained_model)
```

### 9.4 加载已训练好的模型参数


```python
# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
state_dict = paddle.load("./ernie_gram_zh_pointwise_matching_model/model_state.pdparams")

# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
# state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)
```

### 9.5 开始预测


```python
for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)
```


```python
# 执行预测函数
y_probs = predict(model, predict_data_loader)

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)
```

### 9.6输出预测结果


```python
# 我们按照千言文本相似度竞赛的提交格式将预测结果存储在 lcqmc.tsv 中，用来后续提交
# 同时将预测结果输出到终端，便于大家直观感受模型预测效果

test_ds = load_dataset("lcqmc", splits=["test"])

with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
    f.write("index\tprediction\n")    
    for idx, y_pred in enumerate(y_preds):
        f.write("{}\t{}\n".format(idx, y_pred))
        text_pair = test_ds[idx]
        text_pair["label"] = y_pred
        print(text_pair)
```

### 9.7提交 lcqmc 预测结果[千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)

千言文本相似度竞赛一共有 3 个数据集: lcqmc、bq_corpus、paws-x, 我们刚才生成了 lcqmc 的预测结果 lcqmc.tsv, 同时我们在项目内提供了 bq_corpus、paw-x 数据集的空预测结果，我们将这 3 个文件打包提交到千言文本相似度竞赛，即可看到自己的模型在 Lcqmc 数据集上的竞赛成绩。




```python
# 打包预测结果
!zip submit.zip lcqmc.tsv paws-x.tsv bq_corpus.tsv
```

##### 提交预测结果 submit.zip 到 [千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)
### 基线竟然如此简单

# 三、内置 PaddleNLP Datasets API(妈妈说我再也不缺数据集了)

PaddleNLP提供了以下数据集的快速读取API，实际使用时请根据需要**添加splits信息**：

## 1.阅读理解

|  数据集名称   | 简介 | 调用方法 |
|  ----  | ----- | ------ |
|  [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | 斯坦福问答数据集，包括SQuAD1.1和SQuAD2.0|`paddlenlp.datasets.load_dataset('squad')` |
|  [DuReader-yesno](https://aistudio.baidu.com/aistudio/competition/detail/49) | 千言数据集：阅读理解，判断答案极性|`paddlenlp.datasets.load_dataset('dureader_yesno')` |
|  [DuReader-robust](https://aistudio.baidu.com/aistudio/competition/detail/49) | 千言数据集：阅读理解，答案原文抽取|`paddlenlp.datasets.load_dataset('dureader_robust')` |
|  [CMRC2018](http://hfl-rc.com/cmrc2018/) | 第二届“讯飞杯”中文机器阅读理解评测数据集|`paddlenlp.datasets.load_dataset('cmrc2018')` |
|  [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) | 台達閱讀理解資料集|`paddlenlp.datasets.load_dataset('drcd')` |

##  2.文本分类

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [CoLA](https://nyu-mll.github.io/CoLA/) | 单句分类任务，二分类，判断句子是否合法| `paddlenlp.datasets.load_dataset('glue','cola')`|
|  [SST-2](https://nlp.stanford.edu/sentiment/index.html) | 单句分类任务，二分类，判断句子情感极性| `paddlenlp.datasets.load_dataset('glue','sst-2')`|
|  [MRPC](https://microsoft.com/en-us/download/details.aspx?id=52398) | 句对匹配任务，二分类，判断句子对是否是相同意思| `paddlenlp.datasets.load_dataset('glue','mrpc')`|
|  [STSB](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) | 计算句子对相似性，分数为1~5| `paddlenlp.datasets.load_dataset('glue','sts-b')`|
|  [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) | 判定句子对是否等效，等效、不等效两种情况，二分类任务| `paddlenlp.datasets.load_dataset('glue','qqp')`|
|  [MNLI](http://www.nyu.edu/projects/bowman/multinli/) | 句子对，一个前提，一个是假设。前提和假设的关系有三种情况：蕴含（entailment），矛盾（contradiction），中立（neutral）。句子对三分类问题| `paddlenlp.datasets.load_dataset('glue','mnli')`|
|  [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) | 判断问题（question）和句子（sentence）是否蕴含，蕴含和不蕴含，二分类| `paddlenlp.datasets.load_dataset('glue','qnli')`|
|  [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) | 判断句对是否蕴含，句子1和句子2是否互为蕴含，二分类任务| `paddlenlp.datasets.load_dataset('glue','rte')`|
|  [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | 判断句子对是否相关，相关或不相关，二分类任务| `paddlenlp.datasets.load_dataset('glue','wnli')`|
|  [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) | A Large-scale Chinese Question Matching Corpus 语义匹配数据集| `paddlenlp.datasets.load_dataset('lcqmc')`|
|  [ChnSentiCorp](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb) | 中文评论情感分析语料| `paddlenlp.datasets.load_dataset('chnsenticorp')`|


## 3.序列标注

|  数据集名称   | 简介 | 调用方法 |
|  ----  | --------- | ------ |
|  [MSRA_NER](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra) | MSRA 命名实体识别数据集| `paddlenlp.datasets.load_dataset('msra_ner')`|
|  [People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) | 人民日报命名实体识别数据集| `paddlenlp.datasets.load_dataset('peoples_daily_ner')`|


## 4.机器翻译

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [IWSLT15](https://workshop2015.iwslt.org/) | IWSLT'15 English-Vietnamese data 英语-越南语翻译数据集| `paddlenlp.datasets.load_dataset('iwslt15')`|
|  [WMT14ENDE](http://www.statmt.org/wmt14/translation-task.html) | WMT14 EN-DE 经过BPE分词的英语-德语翻译数据集| `paddlenlp.datasets.load_dataset('wmt14ende')`|


## 5.机器同传

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [BSTC](https://aistudio.baidu.com/aistudio/competition/detail/44/) | 千言数据集：机器同传，包括transcription_translation和asr | `paddlenlp.datasets.load_dataset('bstc', 'asr')`|


## 6.文本生成

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [Poetry](https://github.com/chinese-poetry/chinese-poetry) | 中文诗歌古典文集数据| `paddlenlp.datasets.load_dataset('poetry')`|
|  [Couplet](https://github.com/v-zich/couplet-clean-dataset) | 中文对联数据集| `paddlenlp.datasets.load_dataset('couplet')`|

## 7.语料库

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [PTB](http://www.fit.vutbr.cz/~imikolov/rnnlm/) | Penn Treebank Dataset | `paddlenlp.datasets.load_dataset('ptb')`|
|  [Yahoo Answer 100k](https://arxiv.org/pdf/1702.08139.pdf)  | 从Yahoo Answer采样100K| `paddlenlp.datasets.load_dataset('yahoo_answer_100k')`|


# 四、PaddleNLP data API
## 1.APIl列表

| API                             | 简介                                       |
| ------------------------------- | :----------------------------------------- |
| `paddlenlp.data.Stack`          | 堆叠N个具有相同shape的输入数据来构建一个batch |
| `paddlenlp.data.Pad`            | 堆叠N个输入数据来构建一个batch，每个输入数据将会被padding到N个输入数据中最大的长度 |
| `paddlenlp.data.Tuple`          | 将多个batchify函数包装在一起，组成tuple      |
| `paddlenlp.data.Dict`           | 将多个batchify函数包装在一起，组成dict       |
| `paddlenlp.data.SamplerHelper`  | 构建用于`Dataloader`的可迭代sampler         |
| `paddlenlp.data.Vocab`          | 用于文本token和ID之间的映射                  |
| `paddlenlp.data.JiebaTokenizer` | Jieba分词                                  |

## 2.API使用方法

以上API都是用来辅助构建`DataLoader`，`DataLoader`比较重要的三个初始化参数是`dataset`、`batch_sampler`和`collate_fn`。

`paddlenlp.data.Vocab`和`paddlenlp.data.JiebaTokenizer`用在构建`dataset`时处理文本token到ID的映射。

`paddlenlp.data.SamplerHelper`用于构建可迭代的`batch_sampler`。

`paddlenlp.data.Stack`、`paddlenlp.data.Pad`、`paddlenlp.data.Tuple`和`paddlenlp.data.Dict`用于构建生成mini-batch的`collate_fn`函数。

### 2.1数据预处理

#### 2.1.1 `paddlenlp.data.Vocab`

`paddlenlp.data.Vocab`词表类，集合了一系列文本token与ids之间映射的一系列方法，支持从文件、字典、json等一系方式构建词表。

```python
from paddlenlp.data import Vocab
# 从文件构建
vocab1 = Vocab.load_vocabulary(vocab_file_path)
# 从字典构建
# dic = {'unk':0, 'pad':1, 'bos':2, 'eos':3, ...}
vocab2 = Vocab.from_dict(dic)
# 从json构建，一般是已构建好的Vocab对象先保存为json_str或json文件后再进行恢复
# json_str方式
json_str = vocab1.to_json()
vocab3 = Vocab.from_json(json_str)
# json文件方式
vocab1.to_json(json_file_path)
vocab4 = Vocab.from_json(json_file_path)
```

#### 2.1.2 `paddlenlp.data.JiebaTokenizer`

`paddlenlp.data.JiebaTokenizer`初始化需传入`paddlenlp.data.Vocab`类，包含`cut`分词方法和将句子明文转换为ids的`encode`方法。

```python
from paddlenlp.data import Vocab, JiebaTokenizer
# 词表文件路径，运行示例程序可先下载词表文件
# wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
vocab_file_path = './senta_word_dict.txt'
# 构建词表
vocab = Vocab.load_vocabulary(
    vocab_file_path,
    unk_token='[UNK]',
    pad_token='[PAD]')
tokenizer = JiebaTokenizer(vocab)
tokens = tokenizer.cut('我爱你中国') # ['我爱你', '中国']
ids = tokenizer.encode('我爱你中国') # [1170578, 575565]
```

### 2.2 构建`Sampler`

#### 2.2.1 `paddlenlp.data.SamplerHelper`

`paddlenlp.data.SamplerHelper`的作用是构建用于`DataLoader`的可迭代采样器，它包含`shuffle`、`sort`、`batch`、`shard`等一系列方法，方便用户灵活使用。

```python
from paddlenlp.data import SamplerHelper
from paddle.io import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.data = [
            [[1, 2, 3, 4], [1]],
            [[5, 6, 7], [0]],
            [[8, 9], [1]],
        ]

    def __getitem__(self, index):
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label

    def __len__(self):
        return len(self.data)

dataset = MyDataset()
# SamplerHelper返回的是数据索引的可迭代对象，产生的迭代的索引为：[0, 1, 2]
sampler = SamplerHelper(dataset)
# `shuffle()`的作用是随机打乱索引顺序，产生的迭代的索引为：[0, 2, 1]
sampler = sampler.shuffle()
# sort()的作用是按照指定key为排序方式并在buffer_size大小个样本中排序
# 示例中以样本第一个字段的长度进行升序排序，产生的迭代的索引为：[2, 0, 1]
key = (lambda x, data_source: len(data_source[x][0]))
sampler = sampler.sort(key=key, buffer_size=2)
# batch()的作用是按照batch_size组建mini-batch，产生的迭代的索引为：[[2, 0], [1]]
sampler = sampler.batch(batch_size=2)
# shard()的作用是为多卡训练切分数据集，当前卡产生的迭代的索引为：[[2, 0]]
sampler = sampler.shard(num_replicas=2)
```

### 2.3 构建`collate_fn`

#### 2.3.1 `paddlenlp.data.Stack`

`paddlenlp.data.Stack`用来组建batch，其输入必须具有相同的shape，输出便是这些输入的堆叠组成的batch数据。

```python
from paddlenlp.data import Stack
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
"""
[[1, 2, 3, 4],
 [3, 4, 5, 6],
 [5, 6, 7, 8]]
"""
```

#### 2.3.2  `paddlenlp.data.Pad`

`paddlenlp.data.Pad`用来组建batch，它的输入长度不同，它首先会将输入数据全部padding到最大长度，然后再堆叠组成batch数据输出。

```python
from paddlenlp.data import Pad
a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
"""
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
"""
```

#### 2.3.3 `paddlenlp.data.Tuple`

`paddlenlp.data.Tuple`会将多个组batch的函数包装在一起，组成tuple。

```python
from paddlenlp.data import Stack, Pad, Tuple
data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, label = batchify_fn(data)
"""
ids:
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
label: [[1], [0], [1]]
"""
```

#### 2.3.4 `paddlenlp.data.Dict`

`paddlenlp.data.Dict`会将多个组batch的函数包装在一起，组成dict。

```python
from paddlenlp.data import Stack, Pad, Dict
data = [
        {'labels':[1], 'token_ids':[1, 2, 3, 4]},
        {'labels':[0], 'token_ids':[5, 6, 7]},
        {'labels':[1], 'token_ids':[8, 9]},
       ]
batchify_fn = Dict({'token_ids':Pad(pad_val=0), 'labels':Stack()})
ids, label = batchify_fn(data)
"""
ids:
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
label: [[1], [0], [1]]
"""
```

### 2.4 综合示例

```python
from paddlenlp.data import Vocab, JiebaTokenizer, Stack, Pad, Tuple, SamplerHelper
from paddlenlp.datasets import load_dataset
from paddle.io import DataLoader

# 词表文件路径，运行示例程序可先下载词表文件
# wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
vocab_file_path = './senta_word_dict.txt'
# 构建词表
vocab = Vocab.load_vocabulary(
    vocab_file_path,
    unk_token='[UNK]',
    pad_token='[PAD]')
# 初始化分词器
tokenizer = JiebaTokenizer(vocab)

def convert_example(example):
    text, label = example['text'], example['label']
    ids = tokenizer.encode(text)
    label = [label]
    return ids, label

dataset = load_dataset('chnsenticorp', splits='train')
dataset = dataset.map(convert_example, lazy=True)

pad_id = vocab.token_to_idx[vocab.pad_token]
batchify_fn = Tuple(
    Pad(axis=0, pad_val=pad_id),  # ids
    Stack(dtype='int64')  # label
)

batch_sampler = SamplerHelper(dataset).shuffle().batch(batch_size=16)
data_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

# 测试数据集
for batch in data_loader:
    ids, label = batch
    print(ids.shape, label.shape)
    print(ids)
    print(label)
    break
```



```python

```
