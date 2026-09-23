# %% [markdown]
# ## 二元词模型：只给一个字符，并预测下一个字符
# %%
from turtledemo.chaos import plot
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
# %%
words = open('names.txt', 'r').read().splitlines()
print(len(words))
words[:10]
# %% [markdown]
# ### 先给出word中 二元字符的组合
# %%
## 巧妙的方式去获取二元字符组
## 获取数据集中所有二元字符组的次数
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']  # 添加开始结束字符
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
# %%
sorted(b.items(), key=lambda kv: -kv[1])
# %%
import torch
# %%
N = torch.zeros((27, 27), dtype=torch.int32)
# %%
chars = sorted(list(set(''.join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}

stoi['.'] = 0

itos = {i: s for s, i in stoi.items()}

# %% [markdown]
# ## 把二元组映射到tensor， 行是第一个字符，列是第二个字符
# ## 使用<S> <E> 作为最后的元素展示不太好看， 变换成 . = 0 可视化好看一些， 但 <S> = 0 <E> = 27 同样能够达到目标功能
# %%
for w in words:
    chs = ['.'] + list(w) + ['.']  # 添加开始结束字符
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
# %% [markdown]
# - 使用torch.mutltinomail函数来按照概率权重采样，当得到一群数据的概率分布，如果希望按照这个概率分布来采样这批数据，就可以使用这个函数，
# - 在二元组模型中，当知道前一个字符，通过数据集统计的以该字符开头的二元组统计结果和概率分布是知道，因此可以按照这个权重来采样已实现预测的效果
# %%
P = N.float()

### 注意广播的对齐性质，如果size缺维度时，那么会先向右对齐，在复制元素，keepdim就是保证sum前后的shape的维度不变，避免按行求和时，在进行广播时，误认为为按列求和，导致计算错混
P = P / P.sum(1, keepdim=True)
# %%
for _ in range(100):
    idx = 0
    out = []
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True).item()
        out += list(itos[idx])
        if idx == 0:
            break
    print(''.join(out))
# %% [markdown]
# - 当模型构建之后，如何评价模型的精度，如果寻找损失和能够使损失下降的参数成为首要目标
# - 当前这个二元组模型，使用似然来表示精度最为合适，多个条件同时成立的似然表示，但多个概率相乘必然得到很小的值，不是正常的损失值
# - 因此 使用对数似然求和替代 -> 对数函数是x越大，y越接近0，反之越接近负无穷 -> 因此取负对数符合我们常见的损失的表达方式
# 
# %% [markdown]
# ##  使用pytorch 来实现这个过程
# 
# - 为什么神经网络的二元模型，同样在预测之后也要随机采样，那是因为 训练的参数是下一个字符的概率分布，不是某一个字符的概率值
# - 使用随机采样也是为了保证结果的不唯一性
# %%
# 制作二元组的训练集

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

# Tensor 实例化是 float32 tensor 实例化是 int64
xs = torch.tensor(xs)
ys = torch.tensor(ys)

xs.shape

# %%
# 把输入encoder one-hot（独热编码） 向量
import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()

# %%
W = torch.randn(27, 27, requires_grad=True)

logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(5), ys].log().mean()
# %%
W.grad = None
loss.backward()
print(loss)
### 原地修改叶子变量，会导致计算图出现问题，使用no_grad函数可以告知参数的更新不进入计算图中
### 也可以将待优化参数绑定在优化器中，优化器统一优化
with torch.no_grad():
    W += -50 * W.grad
# %% [markdown]
# ## 基于MLP的字符预测模型构建
# %%
## make_more 引入三个先验字符 预测后一个字符
## 获取数据集中所有二元字符组的次数
ratio= 0.8
train_x = []; train_y = []
val_x = []; val_y = []

all_idx = range(len(words))

import random
train_idx = random.sample(all_idx, int(ratio * len(all_idx)))

pre_block_size = 3

def make_data(chs, bls, X, y):
    i = 0
    while i+bls+1 <= len(chs):
        seg = chs[i:i+bls+1]
        i += 1
        seg = list(map(lambda x:stoi[x],seg))
        X.append(seg[:-1])
        y.append(seg[-1])

for i, w in enumerate(words):
    chs = ['.'] * pre_block_size + list(w) + ['.']  # 添加开始结束字符
    if i in train_idx:
      make_data(chs, pre_block_size, train_x, train_y)
    else:
      make_data(chs, pre_block_size, val_x, val_y)

bs = 32

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

val_x = torch.tensor(val_x)
val_y = torch.tensor(val_y)

print('train', train_x.shape, train_y.shape)
print('val', val_x.shape, val_y.shape)

# %%
## 不需要转换成onehot吗？
import torch.nn.functional as F
# train_x = F.one_hot(train_x, num_classes=27).float()
# train_y = F.one_hot(train_y, num_classes=27).float()

train_x.shape
# %%
def batch_random_sampler(a: torch.Tensor, b:torch.Tensor, batch_size: int, seed: int = 42):
    """
    固定 seed，按 batch_size 顺序遍历整个 tensor。
    每次 yield 出来的 batch 互不重复，直到遍历完整个数据集。
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # 1. 产生一次性打乱的完整索引
    perm = torch.randperm(a.size(0), generator=g)

    # 2. 用打乱后的索引重排 tensor，并使用 torch.split 按 batch_size 切分
    shuffled_a = a[perm]
    shuffled_b = b[perm]

    # split 会自动处理末尾不够 batch_size 的情况
    return list(torch.split(shuffled_a, batch_size)), list(torch.split(shuffled_b, batch_size))
# %%
train_x, train_y = batch_random_sampler(train_x, train_y, bs)
len(train_x) == len(train_y)
# %%
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.W1 = torch.randn(3, 6, requires_grad=True)
        self.B1 = torch.randn(6, requires_grad=True)

        self.W2 = torch.randn(6, 27, requires_grad=True)
        self.B2 = torch.randn(27, requires_grad=True)

    def forward(self, x):
        fla_x = tra_x.flatten(1).float()
        L1 = fla_x @ self.W1 + self.B1
        L2 = L1 @ self.W2 + self.B2
        return L2

    def zero_grad(self):
        self.W1.grad = None; self.B1.grad = None
        self.W2.grad= None; self.B2.grad = None

    def step(self):
        self.W1 += -1.0 * lr * self.W1.grad
        self.B1 += -1.0 * lr * self.B1.grad
        self.W2 += -1.0 * lr * self.W2.grad
        self.B2 += -1.0 * lr * self.B2.grad


lr = 0.01
loss_vec = []
val = []

##
mlp = MLP()
for epoch in range(10):
    loss_epoch = 0
    for tra_x, label in zip(train_x, train_y):

        L2 = mlp(tra_x)
        loss = F.cross_entropy(L2, label)
        loss_vec.append(loss.item())

        mlp.zero_grad()
        loss.backward()
        with torch.no_grad():
            mlp.step()
# %%
# %matplotlib inline

plt.plot(np.arange(len(loss_vec)), loss_vec)
plt.show()
# %%
