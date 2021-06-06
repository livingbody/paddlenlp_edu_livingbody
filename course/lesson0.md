# NLP的初体验

大家好，欢迎来到新手入门课程，本次课程将会带领大家进入"人工智能皇冠上的明珠"--自然语言处理（Natural Language Processing, 以下简称NLP），帮助大家掌握基础理论知识，为后续的课程学习打下夯实的基础。

"NLP是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"

NLP技术支撑起一片浩瀚宇宙，下面这些我们熟知的应用背后，都有NLP的身影。

**- 分词、词性标注、命名实体识别**  
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/957d8499cd4a42f38cd2b86516bc96ed1e73f661405b4fef822465eb65dd5734"  width="650" height="400"  ></center>


**- 好评/差评判别、情感分析、舆情分析**   
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ac82a46d9bdc43f4a0203939b7a2b5c8a83932943a0140a5a1b4eded0e23c4f8" width="650" height="200" ></center>


**- 快递单信息自动识别** 
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/d1a54ff20b1c4721b7cd5ad9c63202ed002a40f491ca4e6bb7f2fb2e579ed578" width="650" height="700" ></center>


**- 搜索**  
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/d326771dbcce4ede96259bc224371b501ffcf010267e43b692778846545f2599" width="650" height="600" ></center>


**- 智能问答和对话**   
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/86620a703511470f80d8cf46579dec9dae0f51ba542f4ac18601865d3ef9ac06" width="650" height="600" ></center>


**- 机器同传**   
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/1a531c6fe4ac4d6498c3ef25ee82f9288f3ce86672b744c28ba8888ebec966fb" width="650" height="600" ></center>

以上应用，PaddleNLP中均有实现。本次打卡课程也配套了基于飞桨PaddleNLP的代码实践，方便各位小伙伴快速上手，彻底get NLP技能。

GitHub链接：[https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)    

Gitee链接：[https://gitee.com/paddlepaddle/PaddleNLP](https://gitee.com/paddlepaddle/PaddleNLP)  

**下面是一些NLP领域的重要理论知识点，你不必在这次预习中弄懂，因为在即将开始的打卡营中，我们都会学到这些内容。只有一点点初步的认识就够啦！**

## 一、词向量
**简介**

在自然语言处理任务中，首先需要考虑字、词如何在计算机中表示。通常，有两种表示方式：one-hot表示和分布式表示

+ one-hot表示  
把每个词表示为一个长向量。这个向量的维度是词表大小，向量中只有一个维度的值为1，其余维度为0，这个维度就代表了当前的词。
例如：苹果 [0,0,0,1,0,0,0,0,···]
。one-hot表示不能展示词与词之间的关系，且特征空间非常大。

+ 分布式表示

word embedding指的是将词转化成一种分布式表示，又称词向量。分布式表示将词表示成一个定长的连续的稠密向量。

分布式表示优点:

(1)词之间存在相似关系：是词之间存在“距离”概念，这对很多自然语言处理的任务非常有帮助。

(2)包含更多信息：词向量能够包含更多信息，并且每一维都有特定的含义。在采用one-hot特征时，可以对特征向量进行删减，词向量则不能


### 1. word2vec
在自然语言处理领域，使用上下文描述一个词语的语义是一个常见且有效的做法。2013年，Mikolov提出的经典word2vec算法就是通过上下文来学习语义信息。word2vec包含两个经典模型：CBOW（Continuous Bag-of-Words）和Skip-gram，如**图1**所示。

- **CBOW**：通过上下文的词向量推理中心词。
- **Skip-gram**：根据中心词推理上下文。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/87b90136eef04d7285803e567a5f7f6d0e40bb552deb40a19a7540c4e6aa20a3" width="700" ></center>
<center><br>图1：CBOW和Skip-gram语义学习示意图</br></center>
<br></br>

假设有一个句子“Pineapples are spiked and yellow”，两个模型的推理方式如下：

- 在**CBOW**中，先在句子中选定一个中心词，并把其它词作为这个中心词的上下文。如 **图1** CBOW所示，把“spiked”作为中心词，把“Pineapples、are、and、yellow”作为中心词的上下文。在学习过程中，使用上下文的词向量推理中心词，这样中心词的语义就被传递到上下文的词向量中，如“spiked → pineapple”，从而达到学习语义信息的目的。

- 在**Skip-gram**中，同样先选定一个中心词，并把其他词作为这个中心词的上下文。如 **图1** Skip-gram所示，把“spiked”作为中心词，把“Pineapples、are、and、yellow”作为中心词的上下文。不同的是，在学习过程中，使用中心词的词向量去推理上下文，这样上下文定义的语义被传入中心词的表示中，如“pineapple → spiked”，
从而达到学习语义信息的目的。

+ 论文链接：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 

### 2.  预训练词向量
word2vec之后，涌现了更多word embedding方式，如Glove、fasttext、ElMo等。如今，已有很多预训练完成的词向量，可直接调用使用，用来初始化，可提升简单网络的收敛速度、精度。
+ 代码链接  
[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/embeddings](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/embeddings)

+ AIStudio项目 - 使用预训练词向量优化分类模型效果   
[https://aistudio.baidu.com/aistudio/projectdetail/1535355](https://aistudio.baidu.com/aistudio/projectdetail/1535355)

## 二、RNN和CNN网络

循环神经网络(Recurrent Neural Networks，RNN)已经在众多NLP任务中取得了巨大成功以及广泛应用。RNN的目的使用来处理序列数据。在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。而在RNN网络中，一个序列当前的输出与前面的信息也有关，网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点是有连接的。理论上，RNNs能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关。

如**图2**是一个RNN网络结构，   
输入：一个序列信息，如一句话   
运行：从左到右逐词处理，不断调用一个相同的网络单元  
t是时刻，x是输入层，s是隐藏层，o是输出层，矩阵W就是隐藏层上一次的值作为这一次的输入的权重。    
在t时刻，接收到输入𝑥_t，和隐藏层状态𝑠_(t−1)，得到新的隐藏层状态𝑠_t 和输出𝑜_t。后面的输出依赖于前文。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e416aa610a7c4c02b01094d3c7544f2d3e4bd1f519404177a7f1f7a2e524d36b" width="700" ></center>
<center><br>图2：RNN结构示意图</br></center>
<br></br>

在RNN中，目前使用最广泛的模型便是LSTM(Long Short-Term Memory，长短时记忆模型)模型，该模型能够更好地建模长序列。

除RNN外，卷积神经网络（Convolutional Neural Networks，简称CNN）也被用来对文本进行建模。

怎么使用RNN、CNN呢，且看👇🏻

+ 代码链接   
[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn)

+ 基于LSTM的文本分类  
[https://aistudio.baidu.com/aistudio/projectdetail/1283423](https://aistudio.baidu.com/aistudio/projectdetail/1283423)


## 三、 Transformer

Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由Self-Attenion和Feed Forward Neural Network组成。只要计算资源够，可以通过堆叠多层Transformer来搭建复杂网络。

考虑到RNN（或者LSTM，GRU等）类模型只能从左向右依次计算或者从右向左依次计算，带来了一定的局限性：

1、时刻$t$的计算依赖$t-1$时刻的计算结果，这样限制了模型的并行能力；

2、顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象，LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离缩小为一个常量；其次它不无需依次输入序列信息，因此具有更好的并行性，符合现有的GPU框架。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e55e975e5f8b439d90448ea87a44b1f832240f6b078c4faa9676c64501f8d3ae" width="400" height="400" ></center>
<center><br>图3：Transformer网络结构图</br></center>
<br></br>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b6f2f48ad8b64486a770553547f5d72e06653a9faa9a4f07b620f5b0164b6508" width="400" height="220"></center>
<center><br>图4：Multi-Head Attention</br></center>
<br></br>

**如图3**网络由若干相同的layer 堆叠组成，每个layer主要由**图4**多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。

* Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。
* Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 Residual Connection 和 Layer Normalization 来促进梯度传播和模型收敛。

更具体的关于Transformer的理解可以参考博客：[http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

如何使用Transformer，请戳👇🏻

代码链接：  
[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/transformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/transformer)

使用Transformer进行机器翻译项目链接：   
[https://aistudio.baidu.com/aistudio/projectdetail/1918692](https://aistudio.baidu.com/aistudio/projectdetail/1918692)

论文链接：[Attention is All You Need](https://research.google/pubs/pub46201/)

## 四、预训练模型

Transformer结构的出现催生了众多高质量的预训练模型。2018年开始，以BERT为代表的语义表示预训练模型取得了巨大突破，横扫各大NLP任务基准，带来了预训练+细调的NLP技术变革，该技术逐渐成为突破AI认知技术的关键性技术。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/da0af457e8b74ec59c512c60dc0e30a4b8977875eb8845b6abd41f2c82c37aed" width="650" height="400"></center>
<center><br>图5：预训练模型代表</br></center>
<br></br>

<br>

### BERT

<br>
BERT，Bidirectional Encoder Representation from Transformers，其Encoder由双向Transformer构成。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

BERT模型的结构如下图最左：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/05f0b4c261b442ed8272ffbaa61438de0194b194c5e9468685ddb262020a7380" width="700" ></center>
<center><br>图6：预训练模型对比</br></center>
<br></br>

对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接，能更好的关注到上下文信息；对比ELMo，虽然都是双向的，但是Bert采用Transformer抽取特征效果要优于ELMo采用LSTM抽取特征。

Bert的输入由三种Embedding组成，如**图7**所示：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/da233d3d03cd4ae7982b24f99f042ec5037801367591481fba0c39bb83720bfa" width="700" ></center>
<center><br>图7：Embedding输入</br></center>
<br></br>


其中：

+ Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务

+ Segment Embeddings用来区别两种句子，用于Pre-train中的分类任务

Pre-train任务主要有两个，其一是Masked LM，其二是Next Sentence Prediction：

**Masked LM**：

此任务预训练的目标就是语言模型，在训练过程中作者随机mask 15%的token，而不是把像cbow一样把每个词都预测一遍。最终的损失函数只计算被mask掉的那个token。

**Next Sentence Prediction**

因为涉及到QA和NLI之类的任务，增加了第二个预训练任务，目的是让模型理解两个句子之间的联系。训练的输入是句子A和B，B有一半的几率是A的下一句，输入这两个句子，模型预测B是不是A的下一句。预训练的时候可以达到97-98%的准确度。

**注意**：语料的选取很关键，要选用document-level的而不是sentence-level的，这样可以具备抽象连续长序列特征的能力。

如图8所示，通过调整模型的输入输出部分，Bert模型可实现在不同的任务上进行fine-tune。

+ 单句子分类：CLS+句子。利用CLS进行分类
+ 多句子分类：CLS+句子A+SEP+句子B。利用CLS分类
+ SQuAD：CLS+问题+SEP+文章。利用所有文章单词的输出做计算start和end
+ NER：CLS+句子。利用句子单词做标记

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ef253f50c2f14852a676903cfdfa3fa3ce56a5e74d3f43f0a8338a53533de35c" width="700" ></center>
<center><br>图8：Bert模型不同任务微调</br></center>
<br></br>

BERT Pre-Train 代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/legacy/pretrain_language_models/BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/legacy/pretrain_language_models/BERT)

BERT Fine-tune 下游任务代码链接：[https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/bert/modeling.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/bert/modeling.py)

论文链接：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

<br>

### ERNIE

<br>

ERNIE: Enhanced Language Representation with Informative Entities

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

传统的pre-training 模型主要基于文本中words  和 sentences 之间的共现进行学习，事实上，训练文本数据中的词法结构、语法结构、语义信息也同样是很重要的。命名实体识别中人名、机构名、组织名等名词包含的概念信息对应了词法结构，句子之间的顺序对应了语法结构，文章中的语义相关性对应了语义信息。ERNIE 挖掘训练数据中这些有价值的信息，还可以在大型数据集合中进行增量训练。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/757116f01d924f5d85ac23d6af799a6dbf09068f0f83480db7cdb3cb2ecd125e" width="650" height="700" ></center>


近期，百度文心 ERNIE 最新开源四大预训练模型：多粒度语言知识增强模型 ERNIE-Gram、长文本理解模型 ERNIE-Doc、融合场景图知识的跨模态理解模型 ERNIE-ViL、语言与视觉一体的模型 ERNIE-UNIMO。ERNIE-Gram在多项任务中表现出优异效果。


ERNIE Pre-Train 代码链接：[https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py)

ERNIE Fine-Tune 代码链接
[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/ernie](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/ernie)


使用ERNIE预训练模型完成文本分类 项目链接：[https://aistudio.baidu.com/aistudio/projectdetail/1294333](https://aistudio.baidu.com/aistudio/projectdetail/1294333)  

论文链接：

[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129) 

**如果你读到了这里，那么关闭这个预习课程的动作就是我们的握手礼啦，《基于深度学习的自然语言处理》课程团队欢迎你和我们进行技术探讨，准备，开启你的NLP打卡之旅吧!**
