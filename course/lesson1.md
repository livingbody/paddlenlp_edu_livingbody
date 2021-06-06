# 我的PaddleNLP学习笔记

# 一、PaddleNLP加载预训练词向量

6.7日NLP直播打卡课开始啦

**[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**

**[课程地址请戳这里👈](https://aistudio.baidu.com/aistudio/course/introduce/24177)**

欢迎来课程**QQ群**（群号:618354318）交流吧~~


词向量（Word embedding），即把词语表示成实数向量。“好”的词向量能体现词语直接的相近关系。词向量已经被证明可以提高NLP任务的性能，例如语法分析和情感分析。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/54878855b1df42f9ab50b280d76906b1e0175f280b0f4a2193a542c72634a9bf" width="60%" height="50%"> <br />
</p>
<br><center>图1：词向量示意图</center></br>



PaddleNLP已预置多个公开的预训练Embedding，您可以通过使用`paddlenlp.embeddings.TokenEmbedding`接口加载预训练Embedding，从而提升训练效果。本篇教程将依次介绍`paddlenlp.embeddings.TokenEmbedding`的初始化和文本表示效果，并通过文本分类训练的例子展示其对训练提升的效果。

# 二、PaddleNLP安装
AI Studio平台后续会默认安装PaddleNLP，在此之前可使用如下命令安装。


```python
!pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```


```python
from functools import partial

import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import Pad, Stack, Tuple, Vocab

import data
```

# 三、PaddleNLP使用



## 1. TokenEmbedding初始化
- `embedding_name`
将模型名称以参数形式传入TokenEmbedding，加载对应的模型。默认为`w2v.baidu_encyclopedia.target.word-word.dim300`的词向量。
- `unknown_token`
未知token的表示，默认为[UNK]。
- `unknown_token_vector`
未知token的向量表示，默认生成和embedding维数一致，数值均值为0的正态分布向量。
- `extended_vocab_path`
扩展词汇列表文件路径，词表格式为一行一个词。如引入扩展词汇列表，trainable=True。
- `trainable`
Embedding层是否可被训练。True表示Embedding可以更新参数，False为不可更新。默认为True。


```python
# 初始化TokenEmbedding， 预训练embedding未下载时会自动下载并加载数据
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)
```

    100%|██████████| 694483/694483 [00:10<00:00, 69402.57it/s]
    [2021-06-06 23:03:00,552] [    INFO] - Loading token embedding...
    [2021-06-06 23:03:10,475] [    INFO] - Finish loading embedding vector.
    [2021-06-06 23:03:10,478] [    INFO] - Token Embedding info:             
    Unknown index: 635963             
    Unknown token: [UNK]             
    Padding index: 635964             
    Padding token: [PAD]             
    Shape :[635965, 300]


    Object   type: TokenEmbedding(635965, 300, padding_idx=635964, sparse=False)             
    Unknown index: 635963             
    Unknown token: [UNK]             
    Padding index: 635964             
    Padding token: [PAD]             
    Parameter containing:
    Tensor(shape=[635965, 300], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[-0.24200200,  0.13931701,  0.07378800, ...,  0.14103900,  0.05592300, -0.08004800],
            [-0.08671700,  0.07770800,  0.09515300, ...,  0.11196400,  0.03082200, -0.12893000],
            [-0.11436500,  0.12201900,  0.02833000, ...,  0.11068700,  0.03607300, -0.13763499],
            ...,
            [ 0.02628800, -0.00008300, -0.00393500, ...,  0.00654000,  0.00024600, -0.00662600],
            [ 0.00743478, -0.00040147,  0.00931276, ..., -0.01128159, -0.00069775, -0.00615075],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])


## 2. TokenEmbedding.search
获得指定词汇的词向量。


```python
test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)
```

    [[ 0.260801  0.1047    0.129453 -0.257317 -0.16152   0.19567  -0.074868
       0.361168  0.245882 -0.219141 -0.388083  0.235189  0.029316  0.154215
      -0.354343  0.017746  0.009028  0.01197  -0.121429  0.096542  0.009255
       0.039721  0.363704 -0.239497 -0.41168   0.16958   0.261758  0.022383
      -0.053248 -0.000994 -0.209913 -0.208296  0.197332 -0.3426   -0.162112
       0.134557 -0.250201  0.431298  0.303116  0.517221  0.243843  0.022219
      -0.136554 -0.189223  0.148563 -0.042963 -0.456198  0.14546  -0.041207
       0.049685  0.20294   0.147355 -0.206953 -0.302796 -0.111834  0.128183
       0.289539 -0.298934 -0.096412  0.063079  0.324821 -0.144471  0.052456
       0.088761 -0.040925 -0.103281 -0.216065 -0.200878 -0.100664  0.170614
      -0.355546 -0.062115 -0.52595  -0.235442  0.300866 -0.521523 -0.070713
      -0.331768  0.023021  0.309111 -0.125696  0.016723 -0.0321   -0.200611
       0.057294 -0.128891 -0.392886  0.423002  0.282569 -0.212836  0.450132
       0.067604 -0.124928 -0.294086  0.136479  0.091505 -0.061723 -0.577495
       0.293856 -0.401198  0.302559 -0.467656  0.021708 -0.088507  0.088322
      -0.015567  0.136594  0.112152  0.005394  0.133818  0.071278 -0.198807
       0.043538  0.116647 -0.210486 -0.217972 -0.320675  0.293977  0.277564
       0.09591  -0.359836  0.473573  0.083847  0.240604  0.441624  0.087959
       0.064355 -0.108271  0.055709  0.380487 -0.045262  0.04014  -0.259215
      -0.398335  0.52712  -0.181298  0.448978 -0.114245 -0.028225 -0.146037
       0.347414 -0.076505  0.461865 -0.105099  0.131892  0.079946  0.32422
      -0.258629  0.05225   0.566337  0.348371  0.124111  0.229154  0.075039
      -0.139532 -0.08839  -0.026703 -0.222828 -0.106018  0.324477  0.128269
      -0.045624  0.071815 -0.135702  0.261474  0.297334 -0.031481  0.18959
       0.128716  0.090022  0.037609 -0.049669  0.092909  0.0564   -0.347994
      -0.367187 -0.292187  0.021649 -0.102004 -0.398568 -0.278248 -0.082361
      -0.161823  0.044846  0.212597 -0.013164  0.005527 -0.004024  0.176243
       0.237274 -0.174856 -0.197214  0.150825 -0.164427 -0.244255 -0.14897
       0.098907 -0.295891 -0.013408 -0.146875 -0.126049  0.033235 -0.133444
      -0.003258  0.082053 -0.162569  0.283657  0.315608 -0.171281 -0.276051
       0.258458  0.214045 -0.129798 -0.511728  0.198481 -0.35632  -0.186253
      -0.203719  0.22004  -0.016474  0.080321 -0.463004  0.290794 -0.003445
       0.061247 -0.069157 -0.022525  0.13514   0.001354  0.011079  0.014223
      -0.079145 -0.41402  -0.404242 -0.301509  0.036712  0.037076 -0.061683
      -0.202429  0.130216  0.054355  0.140883 -0.030627 -0.281293 -0.28059
      -0.214048 -0.467033  0.203632 -0.541544  0.183898 -0.129535 -0.286422
      -0.162222  0.262487  0.450505  0.11551  -0.247965 -0.15837   0.060613
      -0.285358  0.498203  0.025008 -0.256397  0.207582  0.166383  0.669677
      -0.067961 -0.049835 -0.444369  0.369306  0.134493 -0.080478 -0.304565
      -0.091756  0.053657  0.114497 -0.076645 -0.123933  0.168645  0.018987
      -0.260592 -0.019668 -0.063312 -0.094939  0.657352  0.247547 -0.161621
       0.289043 -0.284084  0.205076  0.059885  0.055871  0.159309  0.062181
       0.123634  0.282932  0.140399 -0.076253 -0.087103  0.07262 ]]



```python
print(test_token_embedding.shape)
```

    (1, 300)


## 3.TokenEmbedding.cosine_sim
计算词向量间余弦相似度，语义相近的词语余弦相似度更高，说明预训练好的词向量空间有很好的语义表示能力。


```python
score1 = token_embedding.cosine_sim("女孩", "女人")
score2 = token_embedding.cosine_sim("女孩", "书籍")
score3 = token_embedding.cosine_sim("女孩", "男孩")
score4 = token_embedding.cosine_sim("中国", "美国")
score5 = token_embedding.dot("中国", "美国")
print('score1:',score1)
print('score2:',score2)
print('score3:',score3)
print('score4:',score4)
print('score5:',score5)
```

    score1: 0.7017183
    score2: 0.19189896
    score3: 0.7981167
    score4: 0.49586025
    score5: 8.611071


## 4.词向量映射到低维空间

使用深度学习可视化工具[VisualDL](https://github.com/PaddlePaddle/VisualDL)的[High Dimensional](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#High-Dimensional--%E6%95%B0%E6%8D%AE%E9%99%8D%E7%BB%B4%E7%BB%84%E4%BB%B6)组件可以对embedding结果进行可视化展示，便于对其直观分析，步骤如下：

1. 由于AI Studio当前支持的是VisualDL 2.1版本，因此需要升级到2.2版本体验最新的数据降维功能

`pip install --upgrade visualdl`

2. 创建LogWriter并将记录词向量
3. 点击左侧面板中的可视化tab，选择‘hidi’作为文件并启动VisualDL可视化


```python
!pip install --upgrade visualdl
```


```python
# 获取词表中前1000个单词
labels = token_embedding.vocab.to_tokens(list(range(0,1000)))
test_token_embedding = token_embedding.search(labels)

# 引入VisualDL的LogWriter记录日志
from visualdl import LogWriter

with LogWriter(logdir='./hidi') as writer:
    #writer.add_embeddings(tag='test', mat=test_token_embedding, metadata=labels)
    writer.add_embeddings(tag='test', mat=[i for i in test_token_embedding], metadata=labels)
```

### 启动VisualDL查看词向量降维效果
启动步骤：
- 1、切换到「可视化」指定可视化日志
- 2、日志文件选择 'hidi'
- 3、点击「启动VisualDL」后点击「打开VisualDL」，选择「高维数据映射」，即可查看词表中前1000词UMAP方法下映射到三维空间的可视化结果:

![](https://user-images.githubusercontent.com/48054808/120594172-1fe02b00-c473-11eb-9df1-c0206b07e948.gif)

可以看出，语义相近的词在词向量空间中聚集(如数字、章节等)，说明预训练好的词向量有很好的文本表示能力。

使用VisualDL除可视化embedding结果外，还可以对标量、图片、音频等进行可视化，有效提升训练调参效率。关于VisualDL更多功能和详细介绍，可参考[VisualDL使用文档](https://github.com/PaddlePaddle/VisualDL/tree/develop/docs)。

# 四、文本分类任务
以下通过文本分类训练的例子展示paddlenlp.embeddings.TokenEmbedding对训练提升的效果。
## 1.数据准备
### 1.1下载词汇表文件dict.txt，用于构造词-id映射关系。

data.py
```
import numpy as np
import paddle
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
import jieba
tokenizer = jieba


def set_tokenizer(vocab):
    global tokenizer
    if vocab is not None:
        tokenizer = JiebaTokenizer(vocab=vocab)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def convert_tokens_to_ids(tokens, vocab):
    """ Converts a token id (or a sequence of id) in a token string
        (or a sequence of tokens), using the vocabulary.
    """

    ids = []
    unk_id = vocab.get('[UNK]', None)
    for token in tokens:
        wid = vocab.get(token, unk_id)
        if wid:
            ids.append(wid)
    return ids


def convert_example(example, vocab, unk_token_id=1, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks. 
    It use `jieba.cut` to tokenize text.
    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        vocab(obj:`dict`): The vocabulary.
        unk_token_id(obj:`int`, defaults to 1): The unknown token id.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
    Returns:
        input_ids(obj:`list[int]`): The list of token ids.s
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    input_ids = []
    for token in tokenizer.cut(example['text']):
        token_id = vocab.get(token, unk_token_id)
        input_ids.append(token_id)
    valid_length = len(input_ids)

    if not is_test:
        label = np.array(example['label'], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def pad_texts_to_max_seq_len(texts, max_seq_len, pad_token_id=0):
    """
    Padded the texts to the max sequence length if the length of text is lower than it.
    Unless it truncates the text.
    Args:
        texts(obj:`list`): Texts which contrains a sequence of word ids.
        max_seq_len(obj:`int`): Max sequence length.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.
    """
    for index, text in enumerate(texts):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [pad_token_id for _ in range(max_seq_len - seq_len)]
            new_text = text + padded_tokens
            texts[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            texts[index] = new_text

def preprocess_prediction_data(data, vocab):
    """
    It process the prediction data as the format used as training.
    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.
    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
    """
    examples = []
    for text in data:
        tokens = " ".join(tokenizer.cut(text)).split(' ')
        ids = convert_tokens_to_ids(tokens, vocab)
        examples.append([ids, len(ids)])
    return examples

def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      pad_token_id=0):
    """
    Creats dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`False`): Whether to use gpu to run.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn, lazy=True)

    shuffle = True if mode == 'train' else False
    sampler = paddle.io.BatchSampler(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader

```


```python
!wget https://paddlenlp.bj.bcebos.com/data/dict.txt
```

### 1.2 加载词表和数据集，数据集来自PaddleNLP内置的公开中文情感分析数据集ChnSenticorp，一键即可加载。


```python
# Loads vocab.
vocab_path='./dict.txt'  
vocab = data.load_vocab(vocab_path)
if '[PAD]' not in vocab:
    vocab['[PAD]'] = len(vocab)
# Loads dataset.
train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
```

    100%|██████████| 1909/1909 [00:00<00:00, 53805.65it/s]


### 1.3 创建运行和预测时所需要的DataLoader对象。


```python
# Reads data and generates mini-batches.
trans_fn = partial(
    data.convert_example,
    vocab=vocab,
    unk_token_id=vocab['[UNK]'],
    is_test=False)
train_loader = data.create_dataloader(
    train_ds,
    trans_fn=trans_fn,
    batch_size=64,
    mode='train',
    pad_token_id=vocab['[PAD]'])
dev_loader = data.create_dataloader(
    dev_ds,
    trans_fn=trans_fn,
    batch_size=64,
    mode='validation',
    pad_token_id=vocab['[PAD]'])
test_loader = data.create_dataloader(
    test_ds,
    trans_fn=trans_fn,
    batch_size=64,
    mode='test',
    pad_token_id=vocab['[PAD]'])
```

## 2.模型搭建
使用`BOWencoder`搭建一个BOW模型用于文本分类任务。`BOWencoder`输入序列的全部向量，输出一个序列向量的简单加和后的向量来表示序列语义。


```python
class BoWModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 vocab_path,
                 emb_dim=300,
                 hidden_size=128,
                 fc_hidden_size=96,
                 use_token_embedding=True):
        super().__init__()
        if use_token_embedding:
            self.embedder = TokenEmbedding(
                "w2v.baidu_encyclopedia.target.word-word.dim300", extended_vocab_path=vocab_path)
            emb_dim = self.embedder.embedding_dim
        else:
            padding_idx = vocab_size - 1
            self.embedder = nn.Embedding(
                vocab_size, emb_dim, padding_idx=padding_idx)
        self.bow_encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.dropout = nn.Dropout(p=0.3, axis=1)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        summed = self.dropout(summed)
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)
        return logits
```

### 2.1获得样本数据类别标签数


```python
num_classes = len(train_ds.label_list)
```

### 2.2 定义模型
定义一个使用预训练好的Tokenembedding的BOW模型，可通过设置超参数use_token_embedding=False来定义一个随机初始化embedding向量的BOW模型。


```python
use_token_embedding=True
learning_rate=5e-4

model = BoWModel(
    vocab_size=len(vocab),
    num_classes=num_classes,
    vocab_path=vocab_path,
    use_token_embedding=use_token_embedding)
if use_token_embedding:
    vocab = model.embedder.vocab
    data.set_tokenizer(vocab)
    vocab = vocab.token_to_idx
else:
    v = Vocab.from_dict(vocab, unk_token="[UNK]", pad_token="[PAD]")
    data.set_tokenizer(v)

model = paddle.Model(model)
```

    [2021-06-06 00:19:57,125] [    INFO] - Loading token embedding...
    [2021-06-06 00:19:58,623] [    INFO] - Start extending vocab.
    [2021-06-06 00:20:08,291] [    INFO] - Finish extending vocab.
    [2021-06-06 00:20:10,966] [    INFO] - Finish loading embedding vector.
    [2021-06-06 00:20:10,969] [    INFO] - Token Embedding info:             
    Unknown index: 0             
    Unknown token: [UNK]             
    Padding index: 750906             
    Padding token: [PAD]             
    Shape :[750907, 300]


## 3模型配置
调用`model.prepare()`配置模型，如损失函数、优化器。


```python
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)

# Defines loss and metric.
criterion = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, criterion, metric)
```

## 4.模型训练

### 4.1 使用VisualDL观测模型训练过程和模型结果
Paddle2.0中利用model.prepare(), model.fit()等高层接口可以轻松实现训练模型的代码部分，同时在Paddle.Model()接口中提供了Callback类，这里通过VisualDL可视化来直观的对模型训练过程和模型结果进行评估。


```python
# 设置VisualDL路径
log_dir = './use_normal_embedding'
if use_token_embedding:
    log_dir = './use_token_embedding'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
```

### 4.2 调用`model.fit()`一键训练模型


```python
model.fit(train_loader,
    dev_loader,
    epochs=20,
    save_dir='./checkpoints',
    callbacks=callback)
```

```
Epoch 20/20
step  10/150 - loss: 0.2043 - acc: 0.9187 - 63ms/step
step  20/150 - loss: 0.0673 - acc: 0.9375 - 56ms/step
step  30/150 - loss: 0.0824 - acc: 0.9510 - 54ms/step
step  40/150 - loss: 0.1066 - acc: 0.9496 - 53ms/step
step  50/150 - loss: 0.1181 - acc: 0.9531 - 52ms/step
step  60/150 - loss: 0.0444 - acc: 0.9560 - 52ms/step
step  70/150 - loss: 0.0328 - acc: 0.9567 - 51ms/step
step  80/150 - loss: 0.2206 - acc: 0.9574 - 51ms/step
step  90/150 - loss: 0.1328 - acc: 0.9580 - 51ms/step
step 100/150 - loss: 0.0814 - acc: 0.9563 - 51ms/step
step 110/150 - loss: 0.0616 - acc: 0.9572 - 51ms/step
step 120/150 - loss: 0.1595 - acc: 0.9573 - 51ms/step
step 130/150 - loss: 0.0690 - acc: 0.9556 - 51ms/step
step 140/150 - loss: 0.1511 - acc: 0.9561 - 51ms/step
step 150/150 - loss: 0.0406 - acc: 0.9563 - 50ms/step
save checkpoint at /home/aistudio/checkpoints/19
Eval begin...
step 10/19 - loss: 0.5991 - acc: 0.8922 - 56ms/step
step 19/19 - loss: 0.5061 - acc: 0.8867 - 37ms/step
Eval samples: 1200
save checkpoint at /home/aistudio/checkpoints/final
```

## 5.训练结果
启动VisualDL步骤同上，日志文件选择 'use_token_embedding'和'use_normal_embedding'训练过程可视化结果如下：

<div align="center">
	<img src="https://ai-studio-static-online.cdn.bcebos.com/290af9c8fc2442c1b29e94b6232926e3922f6669bb7944409014299a807cdb8c" width="95%">
</div>                                                                                                                                     
图中绿色是使用paddlenlp.embeddings.TokenEmbedding进行的实验，蓝色是使用没有加载预训练模型的Embedding进行的实验。可以看到，使用paddlenlp.embeddings.TokenEmbedding的训练，其验证acc变化趋势上升，并收敛于0.90左右，loss变化趋势下降，收敛后相对平稳，不容易过拟合。而没有使用paddlenlp.embeddings.TokenEmbedding的训练，其验证acc变化趋势向下，并收敛于0.88左右。从示例实验可以观察到，使用paddlenlp.embedding.TokenEmbedding能提升训练效果。

# 五、切词



```python
from paddlenlp.data import JiebaTokenizer
tokenizer = JiebaTokenizer(vocab=token_embedding.vocab)
words = tokenizer.cut("中国人民")
print(words) # ['中国人', '民']

tokens = tokenizer.encode("中国人民")
print(tokens) # [12530, 1334]
```

    ['中国人', '民']
    [12530, 1334]


# 六、embedding constant.py源码分析

## 1. 62种embedding name

该文件中可见Embedding name list，一堆一堆的名称，好多啊啊啊。我数数先

```
from enum import Enum
import os.path as osp

URL_ROOT = "https://paddlenlp.bj.bcebos.com"
EMBEDDING_URL_ROOT = URL_ROOT + "/models/embeddings"

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

EMBEDDING_NAME_LIST = [
    # Word2Vec
    # baidu_encyclopedia
    "w2v.baidu_encyclopedia.target.word-word.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-1.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-2.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-4.dim300",
    "w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300",
    "w2v.baidu_encyclopedia.target.word-ngram.1-3.dim300",
    "w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300",
    "w2v.baidu_encyclopedia.target.word-wordLR.dim300",
    "w2v.baidu_encyclopedia.target.word-wordPosition.dim300",
    "w2v.baidu_encyclopedia.target.bigram-char.dim300",
    "w2v.baidu_encyclopedia.context.word-word.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-1.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-2.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-4.dim300",
    "w2v.baidu_encyclopedia.context.word-ngram.1-2.dim300",
    "w2v.baidu_encyclopedia.context.word-ngram.1-3.dim300",
    "w2v.baidu_encyclopedia.context.word-ngram.2-2.dim300",
    "w2v.baidu_encyclopedia.context.word-wordLR.dim300",
    "w2v.baidu_encyclopedia.context.word-wordPosition.dim300",
    # wikipedia
    "w2v.wiki.target.bigram-char.dim300",
    "w2v.wiki.target.word-char.dim300",
    "w2v.wiki.target.word-word.dim300",
    "w2v.wiki.target.word-bigram.dim300",
    # people_daily
    "w2v.people_daily.target.bigram-char.dim300",
    "w2v.people_daily.target.word-char.dim300",
    "w2v.people_daily.target.word-word.dim300",
    "w2v.people_daily.target.word-bigram.dim300",
    # weibo
    "w2v.weibo.target.bigram-char.dim300",
    "w2v.weibo.target.word-char.dim300",
    "w2v.weibo.target.word-word.dim300",
    "w2v.weibo.target.word-bigram.dim300",
    # sogou
    "w2v.sogou.target.bigram-char.dim300",
    "w2v.sogou.target.word-char.dim300",
    "w2v.sogou.target.word-word.dim300",
    "w2v.sogou.target.word-bigram.dim300",
    # zhihu
    "w2v.zhihu.target.bigram-char.dim300",
    "w2v.zhihu.target.word-char.dim300",
    "w2v.zhihu.target.word-word.dim300",
    "w2v.zhihu.target.word-bigram.dim300",
    # finacial
    "w2v.financial.target.bigram-char.dim300",
    "w2v.financial.target.word-char.dim300",
    "w2v.financial.target.word-word.dim300",
    "w2v.financial.target.word-bigram.dim300",
    # literature
    "w2v.literature.target.bigram-char.dim300",
    "w2v.literature.target.word-char.dim300",
    "w2v.literature.target.word-word.dim300",
    "w2v.literature.target.word-bigram.dim300",
    # siku
    "w2v.sikuquanshu.target.word-word.dim300",
    "w2v.sikuquanshu.target.word-bigram.dim300",
    # Mix-large
    "w2v.mixed-large.target.word-char.dim300",
    "w2v.mixed-large.target.word-word.dim300",
    # GOOGLE NEWS
    "w2v.google_news.target.word-word.dim300.en",
    # GloVe
    "glove.wiki2014-gigaword.target.word-word.dim50.en",
    "glove.wiki2014-gigaword.target.word-word.dim100.en",
    "glove.wiki2014-gigaword.target.word-word.dim200.en",
    "glove.wiki2014-gigaword.target.word-word.dim300.en",
    "glove.twitter.target.word-word.dim25.en",
    "glove.twitter.target.word-word.dim50.en",
    "glove.twitter.target.word-word.dim100.en",
    "glove.twitter.target.word-word.dim200.en",
    # FastText
    "fasttext.wiki-news.target.word-word.dim300.en",
    "fasttext.crawl.target.word-word.dim300.en"
]

```


```python
from paddlenlp.embeddings import constant

print(len(constant.EMBEDDING_NAME_LIST))
for item in constant.EMBEDDING_NAME_LIST:
    print(item)
```

    62
    w2v.baidu_encyclopedia.target.word-word.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-1.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-2.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-4.dim300
    w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300
    w2v.baidu_encyclopedia.target.word-ngram.1-3.dim300
    w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300
    w2v.baidu_encyclopedia.target.word-wordLR.dim300
    w2v.baidu_encyclopedia.target.word-wordPosition.dim300
    w2v.baidu_encyclopedia.target.bigram-char.dim300
    w2v.baidu_encyclopedia.context.word-word.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-1.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-2.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-4.dim300
    w2v.baidu_encyclopedia.context.word-ngram.1-2.dim300
    w2v.baidu_encyclopedia.context.word-ngram.1-3.dim300
    w2v.baidu_encyclopedia.context.word-ngram.2-2.dim300
    w2v.baidu_encyclopedia.context.word-wordLR.dim300
    w2v.baidu_encyclopedia.context.word-wordPosition.dim300
    w2v.wiki.target.bigram-char.dim300
    w2v.wiki.target.word-char.dim300
    w2v.wiki.target.word-word.dim300
    w2v.wiki.target.word-bigram.dim300
    w2v.people_daily.target.bigram-char.dim300
    w2v.people_daily.target.word-char.dim300
    w2v.people_daily.target.word-word.dim300
    w2v.people_daily.target.word-bigram.dim300
    w2v.weibo.target.bigram-char.dim300
    w2v.weibo.target.word-char.dim300
    w2v.weibo.target.word-word.dim300
    w2v.weibo.target.word-bigram.dim300
    w2v.sogou.target.bigram-char.dim300
    w2v.sogou.target.word-char.dim300
    w2v.sogou.target.word-word.dim300
    w2v.sogou.target.word-bigram.dim300
    w2v.zhihu.target.bigram-char.dim300
    w2v.zhihu.target.word-char.dim300
    w2v.zhihu.target.word-word.dim300
    w2v.zhihu.target.word-bigram.dim300
    w2v.financial.target.bigram-char.dim300
    w2v.financial.target.word-char.dim300
    w2v.financial.target.word-word.dim300
    w2v.financial.target.word-bigram.dim300
    w2v.literature.target.bigram-char.dim300
    w2v.literature.target.word-char.dim300
    w2v.literature.target.word-word.dim300
    w2v.literature.target.word-bigram.dim300
    w2v.sikuquanshu.target.word-word.dim300
    w2v.sikuquanshu.target.word-bigram.dim300
    w2v.mixed-large.target.word-char.dim300
    w2v.mixed-large.target.word-word.dim300
    w2v.google_news.target.word-word.dim300.en
    glove.wiki2014-gigaword.target.word-word.dim50.en
    glove.wiki2014-gigaword.target.word-word.dim100.en
    glove.wiki2014-gigaword.target.word-word.dim200.en
    glove.wiki2014-gigaword.target.word-word.dim300.en
    glove.twitter.target.word-word.dim25.en
    glove.twitter.target.word-word.dim50.en
    glove.twitter.target.word-word.dim100.en
    glove.twitter.target.word-word.dim200.en
    fasttext.wiki-news.target.word-word.dim300.en
    fasttext.crawl.target.word-word.dim300.en


## 2.token_embedding


```python
from enum import Enum
import os
import os.path as osp
import numpy as np
import logging

import paddle
import paddle.nn as nn
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import _get_sub_home, MODEL_HOME
from paddlenlp.utils.log import logger
from paddlenlp.data import Vocab, get_idx_from_word
from .constant import EMBEDDING_URL_ROOT, PAD_TOKEN, UNK_TOKEN,\
                      EMBEDDING_NAME_LIST

EMBEDDING_HOME = _get_sub_home('embeddings', parent_home=MODEL_HOME)

__all__ = ['list_embedding_name', 'TokenEmbedding']


def list_embedding_name():
    """
    Lists all names of pretrained embedding models paddlenlp provides.
    """
    return list(EMBEDDING_NAME_LIST)


class TokenEmbedding(nn.Embedding):
    """
    A `TokenEmbedding` can load pre-trained embedding model which paddlenlp provides by
    specifying embedding name. Furthermore, a `TokenEmbedding` can load extended vocabulary
    by specifying extended_vocab_path.

    Args:
        embedding_name (`str`, optional):
            The pre-trained embedding model name. Use `paddlenlp.embeddings.list_embedding_name()` to
            list the names of all embedding models that we provide. 
            Defaults to `w2v.baidu_encyclopedia.target.word-word.dim300`.
        unknown_token (`str`, optional):
            Specifies unknown token.
            Defaults to `[UNK]`.
        unknown_token_vector (`list`, optional):
            To initialize the vector of unknown token. If it's none, use normal distribution to
            initialize the vector of unknown token.
            Defaults to `None`.
        extended_vocab_path (`str`, optional):
            The file path of extended vocabulary.
            Defaults to `None`.
        trainable (`bool`, optional):
            Whether the weight of embedding can be trained.
            Defaults to True.
        keep_extended_vocab_only (`bool`, optional):
            Whether to keep the extended vocabulary only, will be effective only if provides extended_vocab_path.
            Defaults to False.
    """

    def __init__(self,
                 embedding_name=EMBEDDING_NAME_LIST[0],
                 unknown_token=UNK_TOKEN,
                 unknown_token_vector=None,
                 extended_vocab_path=None,
                 trainable=True,
                 keep_extended_vocab_only=False):
        vector_path = osp.join(EMBEDDING_HOME, embedding_name + ".npz")
        if not osp.exists(vector_path):
            # download
            url = EMBEDDING_URL_ROOT + "/" + embedding_name + ".tar.gz"
            get_path_from_url(url, EMBEDDING_HOME)

        logger.info("Loading token embedding...")
        vector_np = np.load(vector_path)
        self.embedding_dim = vector_np['embedding'].shape[1]
        self.unknown_token = unknown_token
        if unknown_token_vector is not None:
            unk_vector = np.array(unknown_token_vector).astype(
                paddle.get_default_dtype())
        else:
            unk_vector = np.random.normal(
                scale=0.02,
                size=self.embedding_dim).astype(paddle.get_default_dtype())
        pad_vector = np.array(
            [0] * self.embedding_dim).astype(paddle.get_default_dtype())
        if extended_vocab_path is not None:
            embedding_table = self._extend_vocab(extended_vocab_path, vector_np,
                                                 pad_vector, unk_vector,
                                                 keep_extended_vocab_only)
            trainable = True
        else:
            embedding_table = self._init_without_extend_vocab(
                vector_np, pad_vector, unk_vector)

        self.vocab = Vocab.from_dict(
            self._word_to_idx, unk_token=unknown_token, pad_token=PAD_TOKEN)
        self.num_embeddings = embedding_table.shape[0]
        # import embedding
        super(TokenEmbedding, self).__init__(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self._word_to_idx[PAD_TOKEN])
        self.weight.set_value(embedding_table)
        self.set_trainable(trainable)
        logger.info("Finish loading embedding vector.")
        s = "Token Embedding info:\
             \nUnknown index: {}\
             \nUnknown token: {}\
             \nPadding index: {}\
             \nPadding token: {}\
             \nShape :{}".format(
            self._word_to_idx[self.unknown_token], self.unknown_token,
            self._word_to_idx[PAD_TOKEN], PAD_TOKEN, self.weight.shape)
        logger.info(s)

    def _init_without_extend_vocab(self, vector_np, pad_vector, unk_vector):
        """
        Constructs index to word list, word to index dict and embedding weight.
        """
        self._idx_to_word = list(vector_np['vocab'])
        self._idx_to_word.append(self.unknown_token)
        self._idx_to_word.append(PAD_TOKEN)
        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)
        # insert unk, pad embedding
        embedding_table = np.append(
            vector_np['embedding'], [unk_vector, pad_vector], axis=0)

        return embedding_table

    def _read_vocab_list_from_file(self, extended_vocab_path):
        # load new vocab table from file
        vocab_list = []
        with open(extended_vocab_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                vocab = line.rstrip("\n").split("\t")[0]
                vocab_list.append(vocab)
        return vocab_list

    def _extend_vocab(self, extended_vocab_path, vector_np, pad_vector,
                      unk_vector, keep_extended_vocab_only):
        """
        Constructs index to word list, word to index dict and embedding weight using
        extended vocab.
        """
        logger.info("Start extending vocab.")
        extend_vocab_list = self._read_vocab_list_from_file(extended_vocab_path)
        extend_vocab_set = set(extend_vocab_list)
        # update idx_to_word
        self._idx_to_word = extend_vocab_list
        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)

        # use the Xavier init the embedding
        xavier_scale = np.sqrt(
            6.0 / float(len(self._idx_to_word) + self.embedding_dim))
        embedding_table = np.random.uniform(
            low=-1.0 * xavier_scale,
            high=xavier_scale,
            size=(len(self._idx_to_word),
                  self.embedding_dim)).astype(paddle.get_default_dtype())

        pretrained_idx_to_word = list(vector_np['vocab'])
        pretrained_word_to_idx = self._construct_word_to_idx(
            pretrained_idx_to_word)
        pretrained_embedding_table = np.array(vector_np['embedding'])

        pretrained_vocab_set = set(pretrained_idx_to_word)
        extend_vocab_set = set(self._idx_to_word)
        vocab_intersection = pretrained_vocab_set & extend_vocab_set
        vocab_subtraction = pretrained_vocab_set - extend_vocab_set

        # assignment from pretrained_vocab_embedding to extend_vocab_embedding
        pretrained_vocab_intersect_index = [
            pretrained_word_to_idx[word] for word in vocab_intersection
        ]
        pretrained_vocab_subtract_index = [
            pretrained_word_to_idx[word] for word in vocab_subtraction
        ]
        extend_vocab_intersect_index = [
            self._word_to_idx[word] for word in vocab_intersection
        ]
        embedding_table[
            extend_vocab_intersect_index] = pretrained_embedding_table[
                pretrained_vocab_intersect_index]
        if not keep_extended_vocab_only:
            for idx in pretrained_vocab_subtract_index:
                word = pretrained_idx_to_word[idx]
                self._idx_to_word.append(word)
                self._word_to_idx[word] = len(self._idx_to_word) - 1

            embedding_table = np.append(
                embedding_table,
                pretrained_embedding_table[pretrained_vocab_subtract_index],
                axis=0)

        if self.unknown_token not in extend_vocab_set:
            self._idx_to_word.append(self.unknown_token)
            self._word_to_idx[self.unknown_token] = len(self._idx_to_word) - 1
            embedding_table = np.append(embedding_table, [unk_vector], axis=0)
        else:
            unk_idx = self._word_to_idx[self.unknown_token]
            embedding_table[unk_idx] = unk_vector

        if PAD_TOKEN not in extend_vocab_set:
            self._idx_to_word.append(PAD_TOKEN)
            self._word_to_idx[PAD_TOKEN] = len(self._idx_to_word) - 1
            embedding_table = np.append(embedding_table, [pad_vector], axis=0)
        else:
            embedding_table[self._word_to_idx[PAD_TOKEN]] = pad_vector

        logger.info("Finish extending vocab.")
        return embedding_table

    def set_trainable(self, trainable):
        """
        Whether or not to set the weights of token embedding to be trainable.

        Args:
            trainable (`bool`):
                The weights can be trained if trainable is set to True, or the weights are fixed if trainable is False.

        """
        self.weight.stop_gradient = not trainable

    def search(self, words):
        """
        Gets the vectors of specifying words.

        Args:
            words (`list` or `str` or `int`): The words which need to be searched.

        Returns:
            `numpy.array`: The vectors of specifying words.

        """
        idx_list = self.get_idx_list_from_words(words)
        idx_tensor = paddle.to_tensor(idx_list)
        return self(idx_tensor).numpy()

    def get_idx_from_word(self, word):
        """
        Gets the index of specifying word by searching word_to_idx dict. 

        Args:
            word (`list` or `str` or `int`): The input token word which we want to get the token index converted from.

        Returns:
            `int`: The index of specifying word.

        """
        return get_idx_from_word(word, self.vocab.token_to_idx,
                                 self.unknown_token)

    def get_idx_list_from_words(self, words):
        """
        Gets the index list of specifying words by searching word_to_idx dict.

        Args:
            words (`list` or `str` or `int`): The input token words which we want to get the token indices converted from.

        Returns:
            `list`: The indexes list of specifying words.

        """
        if isinstance(words, str):
            idx_list = [self.get_idx_from_word(words)]
        elif isinstance(words, int):
            idx_list = [words]
        elif isinstance(words, list) or isinstance(words, tuple):
            idx_list = [
                self.get_idx_from_word(word) if isinstance(word, str) else word
                for word in words
            ]
        else:
            raise TypeError
        return idx_list

    def _dot_np(self, array_a, array_b):
        return np.sum(array_a * array_b)

    def _calc_word(self, word_a, word_b, calc_kernel):
        embeddings = self.search([word_a, word_b])
        embedding_a = embeddings[0]
        embedding_b = embeddings[1]
        return calc_kernel(embedding_a, embedding_b)

    def dot(self, word_a, word_b):
        """
        Calculates the dot product of 2 words. Dot product or scalar product is an
        algebraic operation that takes two equal-length sequences of numbers (usually
        coordinate vectors), and returns a single number.

        Args:
            word_a (`str`): The first word string.
            word_b (`str`): The second word string.

        Returns:
            `Float`: The dot product of 2 words.

        """
        dot = self._dot_np
        return self._calc_word(word_a, word_b, lambda x, y: dot(x, y))

    def cosine_sim(self, word_a, word_b):
        """
        Calculates the cosine similarity of 2 word vectors. Cosine similarity is the
        cosine of the angle between two n-dimensional vectors in an n-dimensional space.

        Args:
            word_a (`str`): The first word string.
            word_b (`str`): The second word string.

        Returns:
            `Float`: The cosine similarity of 2 words.

        """
        dot = self._dot_np
        return self._calc_word(
            word_a, word_b,
            lambda x, y: dot(x, y) / (np.sqrt(dot(x, x)) * np.sqrt(dot(y, y))))

    def _construct_word_to_idx(self, idx_to_word):
        """
        Constructs word to index dict.

        Args:
            idx_to_word ('list'):

        Returns:
            `Dict`: The word to index dict constructed by idx_to_word.

        """
        word_to_idx = {}
        for i, word in enumerate(idx_to_word):
            word_to_idx[word] = i
        return word_to_idx

    def __repr__(self):
        """
        Returns:
            `Str`: The token embedding infomation.

        """
        info = "Object   type: {}\
             \nUnknown index: {}\
             \nUnknown token: {}\
             \nPadding index: {}\
             \nPadding token: {}\
             \n{}".format(
            super(TokenEmbedding, self).__repr__(),
            self._word_to_idx[self.unknown_token], self.unknown_token,
            self._word_to_idx[PAD_TOKEN], PAD_TOKEN, self.weight)
        return info

```

## 3.TokenEmbedding方法
### 3.1 list_embedding_name
### 3.2 TokenEmbedding各类方法

#### 3.2.1 set_trainable 是否将token的权重设置为可训练的
#### 3.2.2 search 获取指定单词的向量
#### 3.2.3 get_idx_from_word 获取通过搜索单词\u到\u idx dict来指定单词的索引
#### 3.2.4 get_idx_list_from_words Gets the index list of specifying words by searching word_to_idx dict.
#### 3.2.5 dot  计算两个单词的点积。点积或标量积是一种代数运算，它采用两个等长的数字序列（通常是坐标向量），并返回单个数字
#### 3.2.6 cosine_sim 计算两个词向量的余弦相似度。余弦相似性是n维空间中两个n维向量之间夹角的余弦。


```python
from paddlenlp.embeddings import token_embedding

# list_embedding_name()获取embedding名称
embedding_list=token_embedding.list_embedding_name()
for item in embedding_list:
    print(item)
```

    w2v.baidu_encyclopedia.target.word-word.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-1.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-2.dim300
    w2v.baidu_encyclopedia.target.word-character.char1-4.dim300
    w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300
    w2v.baidu_encyclopedia.target.word-ngram.1-3.dim300
    w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300
    w2v.baidu_encyclopedia.target.word-wordLR.dim300
    w2v.baidu_encyclopedia.target.word-wordPosition.dim300
    w2v.baidu_encyclopedia.target.bigram-char.dim300
    w2v.baidu_encyclopedia.context.word-word.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-1.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-2.dim300
    w2v.baidu_encyclopedia.context.word-character.char1-4.dim300
    w2v.baidu_encyclopedia.context.word-ngram.1-2.dim300
    w2v.baidu_encyclopedia.context.word-ngram.1-3.dim300
    w2v.baidu_encyclopedia.context.word-ngram.2-2.dim300
    w2v.baidu_encyclopedia.context.word-wordLR.dim300
    w2v.baidu_encyclopedia.context.word-wordPosition.dim300
    w2v.wiki.target.bigram-char.dim300
    w2v.wiki.target.word-char.dim300
    w2v.wiki.target.word-word.dim300
    w2v.wiki.target.word-bigram.dim300
    w2v.people_daily.target.bigram-char.dim300
    w2v.people_daily.target.word-char.dim300
    w2v.people_daily.target.word-word.dim300
    w2v.people_daily.target.word-bigram.dim300
    w2v.weibo.target.bigram-char.dim300
    w2v.weibo.target.word-char.dim300
    w2v.weibo.target.word-word.dim300
    w2v.weibo.target.word-bigram.dim300
    w2v.sogou.target.bigram-char.dim300
    w2v.sogou.target.word-char.dim300
    w2v.sogou.target.word-word.dim300
    w2v.sogou.target.word-bigram.dim300
    w2v.zhihu.target.bigram-char.dim300
    w2v.zhihu.target.word-char.dim300
    w2v.zhihu.target.word-word.dim300
    w2v.zhihu.target.word-bigram.dim300
    w2v.financial.target.bigram-char.dim300
    w2v.financial.target.word-char.dim300
    w2v.financial.target.word-word.dim300
    w2v.financial.target.word-bigram.dim300
    w2v.literature.target.bigram-char.dim300
    w2v.literature.target.word-char.dim300
    w2v.literature.target.word-word.dim300
    w2v.literature.target.word-bigram.dim300
    w2v.sikuquanshu.target.word-word.dim300
    w2v.sikuquanshu.target.word-bigram.dim300
    w2v.mixed-large.target.word-char.dim300
    w2v.mixed-large.target.word-word.dim300
    w2v.google_news.target.word-word.dim300.en
    glove.wiki2014-gigaword.target.word-word.dim50.en
    glove.wiki2014-gigaword.target.word-word.dim100.en
    glove.wiki2014-gigaword.target.word-word.dim200.en
    glove.wiki2014-gigaword.target.word-word.dim300.en
    glove.twitter.target.word-word.dim25.en
    glove.twitter.target.word-word.dim50.en
    glove.twitter.target.word-word.dim100.en
    glove.twitter.target.word-word.dim200.en
    fasttext.wiki-news.target.word-word.dim300.en
    fasttext.crawl.target.word-word.dim300.en


# 七、PaddleNLP更多预训练词向量
PaddleNLP提供61种可直接加载的预训练词向量，训练自多领域中英文语料、覆盖多种经典词向量模型（word2vec、glove、fastText）、涵盖不同维度、不同语料库大小，详见[PaddleNLP Embedding API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/embeddings.md)。

# 八、PaddleNLP 更多项目
 - [seq2vec是什么? 瞧瞧怎么用它做情感分析](https://aistudio.baidu.com/aistudio/projectdetail/1283423)
 - [如何通过预训练模型Fine-tune下游任务](https://aistudio.baidu.com/aistudio/projectdetail/1294333)
 - [使用BiGRU-CRF模型完成快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)
 - [使用预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
 - [使用Seq2Seq模型完成自动对联](https://aistudio.baidu.com/aistudio/projectdetail/1321118)
 - [使用预训练模型ERNIE-GEN实现智能写诗](https://aistudio.baidu.com/aistudio/projectdetail/1339888)
 - [使用PaddleNLP预测新冠疫情病例数](https://aistudio.baidu.com/aistudio/projectdetail/1515548)
 - [使用预训练模型完成阅读理解](https://aistudio.baidu.com/aistudio/projectdetail/1339612)
 - [自定义数据集实现文本多分类任务](https://aistudio.baidu.com/aistudio/projectdetail/1468469)
