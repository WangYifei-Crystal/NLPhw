# NLPhw
DeBERTa Pre-training using MLM
本实验报告是根据我对Masked Level Modelling (MLM)的理解制作的，可能包含不必要/低效/不正确的代码。

Model Training（模型训练）
模型训练有两种类型:预训练pre-training和微调fine-tuning.。 
Pre-Training（预训练）
预训练是在一个大数据集上以一种广义的方式训练你的模型的过程，也就是说，它可以用于任何类型的机器学习问题。
Fine-Tuning（微调）
对模型进行微调仅仅意味着根据给定的问题或特定的任务来训练模型。
Pre-Training vs Fine-Tuning（预训练vs微调）
在预训练中，实际上是在为任何类型的任务训练模型。分类/回归、摘要、命名实体识别(NER)、QnA等。这是因为正在训练模型来学习领域上下文中的数据。如。医疗报告，一些推文，或者事实上的反馈。在微调中，基本上是用预训练模型再训练它来完成一些特定的任务。现在，微调模型已经了解了域上下文中的数据以及该特定任务。如。根据医学报告对疾病进行分类。
Pre-Training Techniques（训练的技术）
可以用两种方式预训练你的模型:
1)MLM(掩模语言建模) 
2)NSP(下一句话预测)
MLM
MLM是一种技术，在这种技术中，你取你的标记化样本，用[MASK]标记替换一些标记，并用它来训练你的模型。然后，模型尝试预测[MASK]token的位置应该是什么，并逐渐开始学习数据。MLM教给这个模型单词之间的关系。
Eg.有一个句子：'Deep Learning is so cool! I love neural networks.'现在用[MASK]标记替换一些单词。
Masked Sentence - 'Deep Learning is so [MASK]! I love [MASK] networks.'
NSP
在NSP中，你用两个句子输入模型，你的模型试图预测第二个句子是否在第一个句子之后。NSP教授模型关于句子之间的长期依赖关系。
