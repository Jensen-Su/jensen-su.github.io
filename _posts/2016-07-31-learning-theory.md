---
layout: post
title: The Learning Theroy
date:   2016-07-30 10:10:00
tags: MachineLearning
subclass: 'post tag-MachineLearning'
categories: 'casper'
cover: 'assets/images/valley_with_ball.png'
navigation: True
logo: 'assets/images/ghost.png'
---

$~$

####1. Openning words

我们知道，在机器学习中，当模型过于简单(数据集很大)时，往往会发生**欠拟合(underfitting)**，也就是模型的学习能力太弱，没有很好地捕获到数据集的共性，当用该模型对新数据进行预测时，就会发生偏差;当模型过于复杂(数据集却很小)时，往往很容易发生**过拟合(overfitting)**，也就是模型学习能力太强，把数据中的扰动误以为是数据集的共同特性，当用该模型进行预测时，就会对扰动(比如噪声)十分敏感。

对此周志华教授的《机器学习》有个很好的例子，比如我们采集了一些树叶作为训练集，尝试对树叶进行学习，使得习得的模型可以预测新样例是不是树叶。如果训练集很大，包含了各种奇形怪状的样子，一个欠拟合的模型可能会把所有绿色的新样例都判定为树叶;如果训练集很小，比如恰好采集的树叶的边缘都具有锯齿，一个过拟合的模型将认为锯齿是所有树叶的共性，并将没有锯齿的树叶认定为不是树叶。


这是欠拟合和过拟合的直观理解。不管发生欠拟合还是过拟合，该模型用于预测新的样例时，都会发生明显的误差，这个误差就是**泛化误差(generalization error)**。实际上，泛化误差可以分为三个部分的叠加。也就是偏差(bias)、方差(variance)、和噪声(noise)。当发生欠拟合，偏差比重上升占据主导地位，当发生过拟合时，方差比重则上升并占据主导地位。

于是引出了第一个问题：

**问题1** 偏差和方差看起来是此消彼长的关系，我们应该对它们进行折衷，那么我们能否对它们进行量化，从而从理论上找到一个最好的折衷点呢？

从上面的讨论中发现，我们最关心的其实是模型对新样例的预测能力，或者说泛化误差，然而我们的模型却是在训练集中习得，所习得的模型对于训练集中的样例的预测误差称为经验误差。那么就有了下面的问题：

**问题2** 在训练集上表现良好的模型面对新样例时表现如何呢？训练误差和泛化误差之间是否存在可以量化的联系？ 

**问题3** 如果这种可量化的关系的存在，是否可以证明？需要什么假设呢？


####2. Premininaries

在对这些问题进行探讨之前，我们先引出学习理论中许多理论所依赖的PAC(probably approximately corrent, 概率近似正确)框架中极为重要的一条假设：

**训练集和测试集中的样本相互独立同分布**

本文假设这一分布为**伯努利分布**，也就是**二分类**问题，得到的结果同样适用于其他分布。

######2.1 Notations

- 分布(伯努利):  $\mathcal{D}$
- 训练集：       $S = \\{(x^{(i)}, y^{(i)}); i = 1, ..., m\\}$
- 训练集大小：   $m$
- 特征向量:      $x^{(i)}$
- 对应标签：     $y^{(i)}$
- 特征向量集：   $x$
- 对应标签集：   $y$
- 假设:          $h$ 或 $h_\theta$,其中$\theta$为假设中待训练的模型参数 
- 训练误差：     $\hat{\epsilon}(h)$ 
- 泛化误差：     $\epsilon (h)$
- 假设空间：     $\mathcal{H}$
- 假设空间大小： $k$
- 置信度:        $1-\delta,~\delta > 0$


**定义1. 训练误差** $\hat{\epsilon}(h) = \frac{1}{m}\sum_{i=1}^{m} 1\\{h(x^{(i)}) \neq y^{(i)}\\}.$ 也就是 训练集中 被假设 $h$ 误分类的样本比例，也称为*经验风险*或者*经验误差。

**定义2. 泛化误差** $\epsilon (h) = P_{(x,y)\sim\mathcal{D}}(h(x) \neq y).$ 也就是当我们从分布$\mathcal{D}$中抽取一个新样例，假设$h$对其误分类的概率。

######2.2 Assumptions

**假设1** 对于给定的假设，我们是通过**经验最小化(EMR)**的方法来选取模型参数：

$$\hat{\theta} = arg~\mathop{min}\limits_\theta\hat{\epsilon}(h_\theta)$$

意思就是选取使得训练误差最小的模型参数。


**假设2** 对于给定的假设空间 $\mathcal{H}$, 我们也通过经验最小化来选出最优假设：

$$\hat{h} = arg~\mathop{min}\limits_{h \in \mathcal{H}}\hat{\epsilon}(h_\theta)$$

意思就是在假设空间里面选取使得训练误差最小的假设。

####3. Conclusion
做完以上的准备工作之后，我们直接给出结论。对于结论的证明以后再补上。

假设我们有一个大小为$k$的假设集 $\mathcal{H} = \{h_1, ..., h_k\}$. 给定训练样本数 $m$, 给定 $\gamma > 0$, 
那么所有的假设$h_i$的泛化误差$\epsilon (h)$ 偏离经验误差 $\hat{\epsilon}(h)$ 均不大于 $\gamma$ 的概率至少为 

$$1 - 2k~exp(-2\gamma^2m)$$

即**一致收敛**(*uniform convergence*)结论. （也可以这样看，至少有一个假设的经验误差偏离泛化误差$\gamma$之外的概率不大于$2k~exp(-2\gamma^2m)$.)

**问题1**： 给定$\gamma$和置信度$1 - \delta$, $m$ 必须取多大才能保证对于训练误差与泛化误差的偏差$\|\epsilon(h)-\hat{\epsilon}(h)\| \le \gamma$？

**结论**: 令$1 - 2k~exp(-2\gamma^2m) \ge 1 - \delta$, 可以求得$m$的一个下界：

$$m \ge \frac{1}{2\gamma^2}log\frac{2k}{\delta}. $$

这告诉我们：

1. 样本数与给定偏差 $\gamma$ 成二次反比，因此样本越大，训练误差与泛化误差相差越小. 
2. **更有指导意义的是，训练集大小$m$的下界与假设空间$\mathcal{H}$的大小只是对数关系.**

为了保证一定的性能，一个算法所要求训练集的大小 $m$ 称为该算法的**取样复杂度(sample complexity)**.

**问题2**： 给定训练集大小$m$和置信度$1-\delta>0$, 训练误差与泛化误差的偏差$\|\epsilon(h)-\hat{\epsilon}(h)\|$ 的上界是多少？


**结论1**: 令$2k~exp(-2\gamma^2m) = \delta$, 可以求得$\gamma$的一个上界：

$$ |\epsilon(h)-\hat{\epsilon}(h)| \le \sqrt{\frac{1}{2m}log\frac{2k}{\delta}}. $$

**注意，这是对于假设空间的所有假设$h\in\mathcal{H}$都成立的**. 那么对于我们根据经验最小化原则选取的假设$\hat{h} = arg~\mathop{min}\limits_{h \in \mathcal{H}}\hat{\epsilon}(h_\theta)$来说会有什么样的上界呢？

我们把问题2的结论中$\gamma$的上界记为$\gamma_0$. 即$ \|\epsilon(h)-\hat{\epsilon}(h)\| \le \gamma_0$. 

对于我们选取的假设$\hat{h}$有$\|\epsilon(\hat{h})-\hat{\epsilon}(\hat{h})\|\le\gamma_0$, 或者 $\epsilon(\hat{h}) \le \hat{\epsilon}(\hat{h}) + \gamma_0$.

对于最优假设$h^\*$, 同样有$\|\hat{\epsilon}(h^\*)-\epsilon(h^\*)\|\le\gamma_0$, 或者 $\hat{\epsilon}(h^\*) \le \epsilon(h^\*) + \gamma_0$.

**结论2**因此对于我们选取的假设有：

$$\epsilon(\hat{h}) \le \hat{\epsilon}(\hat{h}) + \gamma_0 
                    \le \hat{\epsilon}(h^*) + \gamma_0 
                    \le \epsilon(h^*) + \gamma_0 
$$

也就是说在一致收敛的前提下，根据经验最小化原则习得的假设的泛化误差比假设空间里面最好的假设的泛化误差相差不大于$2\gamma_0$!

**定理** 设 $\|\mathcal{H}\|=k$, 给定训练集大小$m$和置信度$1-\delta$, 对于由经验最小化习得的假设，有：

$$\epsilon(\hat{h}) \le \left(\mathop{min}\limits_{h\in\mathcal{H}}\epsilon(h)\right)+ 2\sqrt{\frac{1}{2m}log\frac{2k}{\delta}}$$

这个定理量化了我们讨论的偏差与方差间的折衷问题。例如原来的假设空间为$\mathcal{H}$,如果我们考虑一个更大的假设空间 $\mathcal{H'} \supsetneqq \mathcal{H}$, 那么上式第一项会减小，也就是偏差减小;然而由于假设空间变大，即$k$变大，会使得第二项变大，也就是方差变大。
