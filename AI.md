
## AI 基础篇

### 监督/全监督/半监督/弱监督

### 卷积的作用？

1. 局部连接。比起全连接，局部连接会大大减少网络的参数。
2. 权值共享。参数共享也能减少整体参数量。一个卷积核的参数权重被整张图片共享，不会因为图像内位置的不同而改变卷积核内的参数权重。
3. 下采样。下采样能逐渐降低图像分辨率，使得计算资源耗费变少，加速模型训练，也能有效控制过拟合。

### Norm/BN 的作用：

神经网络因为是一堆矩阵运算（线性变换）和非线性变化（Relu）的堆叠。
如果 W、b 初始化得不合适，或者上一层输出分布本身不稳定，下一层的输入可能均值飘移或方差过大/过小。（特别是在网络初始的阶段）一层层叠加后，分布会逐渐“失控”。这就是所谓的内部协变量偏移。
N 把输入“拉回”到一个合适的范围（大约均值 0，方差 1），这样能保证梯度比较稳定，也可以帮助收敛。（为什么能帮助收敛？）但是对 BN 对小 batch 的效果不好，所以 Trans 里面会用 LayerNorm。

> 顺序：输入 → 卷积/全连接 → BN → ReLU → 下一层 （要先归一化再输入 ReLU）

#### BatchNorm

将当前批次所有数据的具有相同通道索引的特征图划分为一组，每组单独归一化，这样组集合就是（NxC1xHxW）一组。

#### LayerNorm

将当前批次单个数据的所有通道的特征图划分为一组，每组单独归一化，这样组集合就是（N1xCxHxW）一组。

### CNN 的典型输出：

（NxCxHxW）：
batch size x Channel x Hight x Width
(32 x 3 x 1024 x 768)（例子，假设一个 batch 输入 32 张图片）

### 正则化：

#### L2 正则化：

```bash
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
```

这里的 weight decay 实际上就是正则化，作用是防止特别依赖某个权重，惩罚较大的权重。

#### Dropout：

· 训练时随机把一部分神经元“丢掉”（置 0），以概率 𝑝 保留。

· 推理时不丢弃，而是用缩放的权重。

· 本质：引入随机性，让模型不能过度依赖某些神经元 → 起到正则化效果。
可以在全连接层中使用

```bash
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(1024 * 4 * 4, 2048)
        self.bn_fc1 = nn.BatchNorm1d(num_features=2048)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = fc_with_initialize(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(num_features=1024)
        self.dropout_fc2 = nn.Dropout(0.6)
        self.fc3 = fc_with_initialize(1024, self.num_class)
```

· 没有 Dropout 时，某些神经元可能“绑在一起”工作，过分依赖彼此。通过 Dropout 随机屏蔽掉一些神经元，迫使其他神经元也要学到有用的特征。

· 在 Dropout 每一轮训练过程中随机丢失神经元的操作相当于多个模型进行取平均，因此用于预测时具有 vote 的效果。

> 一个神经元同时包含了输入，权重，偏置和其激活函数，所谓神经元激活不激活是由激活函数决定的

### 过拟合与欠拟合

**欠拟合**
· 模型太简单，不能捕捉数据规律，训练集表现就很差。

· 训练误差高，测试误差也高。

原因可能是：

模型容量不足（比如用线性模型拟合非线性数据）。

特征不够（没包含关键特征）。

**过拟合**
· 模型太复杂，在训练集上表现很好，但泛化到测试集很差。
· 训练误差低，测试误差高。

可以使用：
· 增加数据量（收集更多样本 / 数据增强）。

· 加强正则化（L1/L2、Dropout、BN）。

· 提前停止训练 (Early Stopping)。

· 减小模型复杂度。

### 损失函数

### 激活函数

#### Sigmoid

$\sigma(x) = \frac{1}{1+e^{-x}}$

**优点**

· 好像就只有输出范围是 0 到 1 了
**缺点**

· 计算量大（有幂）

· Sigmoid 的导数范围是 0 到 0.25，x 过大和过小都容易让导数接近于 0

· 输出不是 0 均值，导致网络加深会改变数据的分布

#### tanh

$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
**优点**

· 解决了均值不是 0 的问题

· 靠近 0 点时导数值比 sigmoid 大，收敛快一些
**缺点**

· 计算量还是大

· 梯度消失问题依旧存在

#### ReLU（我的个人项目里是这样的）

$\text{ReLU}(x) = \max(0, x)$
**优点**

· 计算速度很快，非线性函数

· 激活输出为正时，导数为 1，缓解了梯度消失

· 为负时，使得神经元变得稀疏，感觉有点像正则化，防止了部分噪声引入。
**缺点**

· ReLU 也不是 zero-centered 的

· 使得神经元部分死亡，让部分神经元可能无法更新

#### Leaky ReLU（CNN 里常用）

$\text{LeakyReLU}(x) = \begin{cases}x, & x \geq 0 \\ \alpha x, & x < 0\end{cases}$
**优点**

· 在 ReLU 基础上防止了神经元死亡问题
**缺点**

· 网络稀疏性更差

· 引入了额外的超参数

· 相对来讲更贴近 0 均值

#### GELU（公式有点复杂）

**优点**

· 兼具稀疏性和概率性

· Trans/BERT 中经常使用

#### Softmax（勉强也算激活函数吧！实在不知道分到哪）

可以把 logit 映射到和为 1 的一个概率上，所以很适合多分类问题。

**不过现代 DL 主要还是追求防止梯度消失和爆炸问题，0 均值只是让收敛速度更慢一些**

### 优化器

**梯度理解**
神经网络的损失函数是一个多元函数，相当于网络在一个很高维度的空间寻找梯度。
一元函数，要么正要么负方向。但在高位空间就是一个找一个方向高维向量了。

在更新时，每个参数都拥有自己的偏导数分量，一起构成某个“最好的梯度向量”，因此更新一次梯度会牵扯到几乎所有参数。（类似于空间中多个向量合成为一个向量，某个更新方向实际上也是有多个参数组成的，有些参数梯度大，有些参数梯度小）

#### 1.SGD

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})
$$

· 最简单最经典
· 收敛慢容易震荡

#### 2. SGD with Momentum

$$
v_t = \beta v_{t-1} + (1-\beta)\nabla_\theta J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

· 更新方向是历史梯度的指数加权平均，而不是单点的梯度。
· 有助于提高收敛速度

#### 3.Adam

AdaGrad：对每个参数都单独设置学习率，并且加入一个历史梯度

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta J(\theta_t)
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta J(\theta_t))^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t
$$

$$
\theta_{t+1} = \theta_t - \eta \Bigg( \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda \theta_t \Bigg)
$$

### 数据增强

数据增强流水线，torchvision 等等（我的自制视频里面有讲到）

### 怎么样定义一个可学习的权重？

```bash
self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
```

## Transformer Attention 和 ViT

注意，因为不需要“生成”，所以传统的vit仅仅是一个encoder，不需要根本的transformer架构里的decoder。
把最重要的说在最前面，ViT原论文中最核心的结论是，当拥有足够多的数据进行预训练的时候，ViT的表现就会超过CNN，突破transformer缺少归纳偏置的限制，可以在下游任务中获得较好的迁移效果。

因为：CNN 在结构里“内置了强先验（inductive bias）”，而 ViT 把这些先验几乎全部交给数据去学。
CNN 在结构上强行假设了很多世界规律：
局部性（locality） 相邻像素更相关
平移等变性（translation equivariance） 同一个纹理不同位置，输出不变，只是位置改变了
层级组合（hierarchy） 边缘 → 纹理 → 部件 → 目标
这些不是学出来的，是写死在卷积算子里的。
反观，attention 的假设是：
“任意 token 都可能和任意 token 有关系”，导致表达能力更强，但问题在于这需要极大数据量才可以支撑起来。

ViT 中流动的张量核心形态是：
Batch×Token×Feature

Input image:        (B, 3, 224, 224)
Patch embedding →   (B, 196, 768)（196个patch，每个patch有16x16x3的像素，对于每个patch，额外进行一次768→768的linear）
CLS token →       (B, 197, 768)（这个cls token类似于一种全局表示，但是transformer原论文里没有提到，当要产生情感提取/分类时就要使用，这个其实是bert里面提到的，vit为了一定程度上对齐bert结构，所以也引入了cls token）

我们推一遍vit里的流程：我们直接使用大矩阵来一次性代替Q，K，V的所有变换，我们只需要在其中reshape出各个部分就行：（以双头为例子）  然后每个Q和K的转置相乘，得到197x197的注意力权重矩阵（当然这里要使用注意力头的维度进行归一化之后再softmax，防止点积数值溢出），197x197的权重矩阵乘上197x384的单头v，最后把两个头的197x384结果concat在一起，重新得到197x768的原始维度。（因此维度在这里是始终固定的）

在这里，197x768的多头注意力输出会sum上仅仅接受过LN的z（初始embedding输入）的残差链接。然后继续norm，之后还会有一次768→768的线性变换。至此，一个vit的encoder block结束。


CNN 中流动的张量核心形态是：
Batch×Channel×Height×Width

例如resnet：

(B, 3, H, W)
↓ conv
(B, 64, H/2, W/2)
↓ conv
(B, 128, H/4, W/4)

### 为什么ViT里用Gelu，而不是cnn里的relu？
ViT 没有声称 GELU 是“为视觉特别设计的”，而是 完全沿用 NLP Transformer（BERT）的 MLP 设计，因此我们得回退到transformer本身的设计哲学上。
ReLU（硬门控）会把负半轴全部截断

GELU（软门控，概率意义），包含一个高斯函数，保留一个神经元，不是看它是不是 >0，
而是看它在噪声下为正的概率。

同时，我们要注意到，transformer里的LN（layerNorm）不适合relu这种引入稀疏性的函数，把特征标准化到 均值 0，导致很多维度会在 0 附近波动
如果你用 ReLU：约 50% 维度被直接砍成 0。

### 这里还需要继续复习batchnorm和layernorm的区别
Batchnorm的设计，会导致训练时小batch的扰动更大。
假设我图片的中间维度是：B x C x H x W, batchnorm会对每一个channel：在B x H x W作为分母的情况下归一化，也就是我在“跨样本（图片），跨空间”维度上归一化。

Layernorm如果在transformer里，则是：在特征维度上归一化，即对于一个197x768的输入（token x dimension），在每一个token的768个特征维度上归一化，使其均值为0，方差为1。

如果硬在cnn里做Layernorm（实际上cnn的结构不适合如此），那也就是在“单样本”上，使用C x H x W做归一化，这会带来非常奇怪的结果：一个像素点的值会被其他通道影响，我的边缘检测的响应有可能会被颜色响应，噪声响应等其他通道的状态抹平，这实际上让通道的检测丢失了独立性。

### 回到问题，为什么transformer使用layernorm？
因为layernorm依旧保证了transformer的设计哲学，各个token是独立的，每个token是一个独立语义单元，token之间的全局关系应该通过注意力模块学习，而不是直接使用norm来强行统一。
假设我们使用batchnorm：唯一一种可能性是：对每一个 feature 维 d，在 batch × token 维度上做 norm。即分母为B x token，即在normalize中全局混合了其他token里的语义结果，而这是奇怪的。

那么，凭什么cnn里就可以使用batchnorm？这不也是混合了吗？
Cnn的语义本质来自于“通道”，我们初始拥有3个通道，分别代表RGB的语义信息，从底层来看，我们假设cnn在第一层之后提取出32个通道C，其中某个通道c的卷积核可能代表“边缘”语义，这个通道对于batch内的所有样本表达的语义都相同，即都是batch样本内的“边缘”，我们并没有把“不同语义”混合在一起，每一次的归一化都只对某个单独的语义通道做，我们并不会把跨通道信息“噪声检测” “颜色检测”等抽象通道融合在一起，因此这是自洽的。  
### ResNet 和瓶颈块结构

什么都不学（参数趋近于 0）”比“精确学出恒等映射”更自然。
解决深层网络的退化问题和梯度消失爆炸问题。

直接把上面一层跳跃连接到这一层的输出，如果自己没学到什么，至少上一层的 x 还在，不会自己“搞砸”

#### 手搓瓶颈块？

```bash
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4  # 输出通道是中间通道的 4 倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1 卷积 降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积 提取特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积 升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 残差支路（可能需要下采样保证维度一致）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut 分支
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

需要注意捷径链接里的 downsample，如果通 道/高宽不一致，需要对应。
通常是以下函数来改变捷径链接的通道：

```bash
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels)
)

```

## 池化层有几种？

## 目标检测与计算机视觉

### 你能手搓 LeNet 吗？

### 各个边缘检测算子的区别？

### 单阶段检测和双阶段

### 各个正确率指标

### 卡尔曼滤波

### YOLOv7 和 YOLOv5 的区别，为什么你使用了 YOLOv7？

## 3D 感知基础

### 激光雷达原理

激光雷达发射高密度的激光束，光束沿直行传播打到物体的表面，然后以相同的方向反射回去（忽略少量光线发生衍射现象），反射回去的光线由光电探测器（光敏传感器）检测收集，结合激光束往返传播的距离与方向信息就可以生成物体的 3D 几何形状。实际在使用过程中，激光发射器置于连续旋转的底座上，从而使得发射的激光束能以不同方向到达物体表面（前、后、左、右）。

### PointRCNN 架构理解

### ICP 配准

### RANSAC

### BEV

### KPConv

# C++算法

## C++与 C 的区别

封装、继承和多态，面向对象。封装隐藏了实现细节，使得代码模块化；派生类可以继承父类的数据和方法，扩展了已经存在的模块，实现了代码重用；多态则是“一个接口，多种实现”，通过派生类重写父类的虚函数，实现了接口的重用。

## 容器 Vector

> 需要 #include <vector> using namespace std;
> 使用.size()访问当前元素个数
> .capacity()实际分配的容量

### 常用 STL 函数

· sort(nums.begin(),nums.end());快速排序，输出的 nums 就是排序后的数组。
· lower_bound(begin,end,x) 找第一个>=x 的位置
· upper_bound(begin,end,x) 找第一个>x 的位置
· reverse(begin, end) 翻转容器的 begin 位置到 end 为止，左边是闭区间右边是开区间（即如果要反转前三个，end 应该是 begin()+3,而不是+2）
· std::max_element(first, last)，返回左闭右开区间的最大元素（min 同理）

## for 循环

· 对 vector 使用 vec.size()来获取长度，实际上判断是小于该长度而不是小于等于
· 对于数组，使用 sizeof(arr)

```bash
· for(int i=0;i<sizeof(xxx);i++)
```

## 矩阵

· 使用嵌套 vector
vector<vector<int>> matrix = {
{1, 2, 3},
{4, 5, 6},
{7, 8, 9}
};

## const 坑点

const 修饰它左边的东西；
const int \*p 👉 值不可改，指针可改

int \* const p 👉 值可改，指针不可改

const int \* const p 👉 值和指针都不可改

· const int& b = a; // 声明一个常量引用，引用常量 a

## leetcode 80 有感

当你需要返回一个合理数组的长度时，你应该就要写到慢的写指针，这样写到最后自然而然就是返回数组长度，非常自然且简洁。

有时候需要跳出“语义”，用算法的角度看问题（我觉得这需要熟练度和一些套路）
指针的定义一定要明确，不能又当写又当标记某个下标。

## sizeof()学习

用来计算一个类型/对象的所占用的内存大小。
在 32 位系统和 64 位系统中，指针可能是不同的内存大小（前者四位字节（相当于 32bit），后者 8 位）
int 一般是 4byte（即 32 位），double 那就是 8byte，long long 也是 8byte，long 是 4byte。

如果使用 sizeof()一个含有多个不同成员的结构体（Struct），那么需要是各个成员的公倍数。（并非，一般来说要往下补齐，让下一个成员从自己占用的字节的倍数开始）
特例：sizeof(EmptyStruct) == 1，即使是空结构体，sizeof（）也是 1。

## 字符串 char

注意 char 回以\0 作为结尾，因此 sizeof 一般会多一个字节，而 strlen 会忽视\0，输出实用长度。

## string

不能使用 strlen，使用.size()
可以使用 isalnum(c)函数，判断一个字符串里是否有字母或数字
然后可以使用 tolower(c)转换为小写

## 哈希容器

> unordered_map<Key, Value>
> 键值对结构，在以上容器里就可以统计某个字符出现的频率（例如定义为 Char,int），其可以把字符全部映射到哈希值，因此可以存储 a-z 之外的字符。

## 回文题！

```bash
class Solution {
public:
    bool isPalindrome(string s) {
        string NewString;
        for (char c : s) {
        if (isalnum(c)) {   // 保留字母和数字
            NewString += tolower(c);
            }
        }
        int i=0;
        int j=NewString.size()-1;
        while(i<j)
        {
            if(NewString[i]!=NewString[j])
            {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
};
```
# 华为机考 / AI 八股复习笔记整理版

> 适用场景：选择题、概念题、快速判断题、公式题。  
> 复习目标：不是把每个知识点讲成教材，而是把“考场上怎么快速识别、怎么不踩坑”整理清楚。

---

## 0. 快速纠错清单：高频易错点

### 0.1 牛顿迭代公式

**正确公式：**

\[
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
\]

不是：

\[
x_1=x_0-\frac{f'(x_0)}{f(x_0)}
\]

**记忆：**  
牛顿法是在当前点用切线逼近函数零点，所以是“函数值除以导数”。

---

### 0.2 混合精度训练中的 Loss Scaling

在 FP16 / 混合精度训练中，**Loss Scaling 主要为了解决梯度下溢（underflow）**。

因为 FP16 可表示的数值范围有限，很多很小的梯度可能直接变成 0。  
Loss Scaling 会先把 loss 放大，使反向传播中的梯度整体变大，避免小梯度在 FP16 中下溢。

**一句话：**  
Loss Scaling 防的是 **FP16 小梯度下溢**，不是为了提升模型精度本身。

---

### 0.3 \(A^TA\) 与正定 / 半正定

对任意矩阵 \(A\)，都有：

\[
A^TA \succeq 0
\]

即 \(A^TA\) 一定是**对称半正定矩阵**。

因为：

\[
x^T A^T A x = (Ax)^T(Ax)=\|Ax\|^2 \ge 0
\]

如果 \(A\) **满列秩**，则：

\[
A^TA \succ 0
\]

即 \(A^TA\) 是**正定矩阵**。

**注意：**

- \(A^TA\)：天然对称半正定；
- 满列秩：进一步正定；
- \(A+A^T\)：一定对称，但**不一定正定**；
- 判断正定时，默认讨论对象应该是**实对称矩阵**。

---

### 0.4 正定矩阵的顺序主子式判别

对实对称矩阵 \(A\)，若其所有**顺序主子式**都大于 0，则 \(A\) 正定。

也就是从左上角开始看：

\[
\Delta_1>0,\quad \Delta_2>0,\quad \cdots,\quad \Delta_n>0
\]

这就是 Sylvester 判据。

**注意：**

- 这个判据用于判断**正定**；
- 一般需要矩阵是**实对称矩阵**；
- 判断半正定不能只简单看顺序主子式，通常要看所有主子式非负或看特征值非负。

---

### 0.5 KL 散度不是严格距离

KL 散度：

\[
D_{KL}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}
\]

它要求输入是概率分布，而不是任意向量。

KL 散度不是严格意义上的距离，因为它：

- 不对称：  
  \[
  D_{KL}(P\|Q)\neq D_{KL}(Q\|P)
  \]
- 不满足三角不等式；
- 对 \(Q_i=0, P_i>0\) 的情况会出现无穷大。

**一句话：**  
KL 是“分布差异度量”，不是数学意义上的距离。

---

### 0.6 蒙特卡洛积分与维度灾难

蒙特卡洛积分的误差收敛速度通常是：

\[
O(N^{-1/2})
\]

这个速度看起来不快，但它的关键优势是：**对维度不敏感**。

传统网格积分 / 黎曼积分在高维时会遭遇严重的维度灾难。  
所以说“蒙特卡洛只适用于低维，高维应使用黎曼积分”是错误的。

**考试判断：**

- 低维、光滑函数：确定性数值积分可能更快；
- 高维积分：蒙特卡洛往往更有优势；
- 蒙特卡洛典型误差：\(O(N^{-1/2})\)。

---

## 1. 计算机基础：显存、数据类型与单位

### 1.1 Byte 与 Bit

\[
1\ \text{Byte}=8\ \text{Bits}
\]

类型名后面的数字通常表示该类型占用多少个 bit：

| 数据类型 | bit 数 | byte 数 |
|---|---:|---:|
| FP16 / INT16 | 16 bit | 2 Byte |
| FP32 / INT32 | 32 bit | 4 Byte |
| FP64 / INT64 | 64 bit | 8 Byte |
| INT8 | 8 bit | 1 Byte |

### 1.2 显存占用的基础公式

如果一个张量有 \(N\) 个元素，每个元素占 \(b\) Byte，那么显存占用约为：

\[
N \times b
\]

例如：

\[
1000 \times 1000 \text{ 的 FP32 矩阵}
\]

元素数量：

\[
10^6
\]

每个 FP32 占 4 Byte，所以显存约为：

\[
4\times 10^6 \text{ Byte}\approx 4\text{ MB}
\]

### 1.3 训练显存通常不只是参数

训练时显存主要来自：

1. 模型参数；
2. 梯度；
3. 优化器状态；
4. 激活值；
5. 临时 buffer；
6. KV cache（推理时更常见）。

Adam 优化器通常会额外保存一阶矩和二阶矩：

\[
m_t,\quad v_t
\]

所以同样参数量下，Adam 的训练显存通常明显高于 SGD。

---

## 2. 概率分布与统计基础

### 2.1 泊松过程与指数分布

如果事件发生服从参数为 \(\lambda\) 的泊松过程，那么任意连续两次事件的时间间隔 \(T\) 服从参数为 \(\lambda\) 的指数分布：

\[
T\sim \text{Exp}(\lambda)
\]

指数分布的密度函数：

\[
f(t)=\lambda e^{-\lambda t},\quad t\ge 0
\]

均值：

\[
E[T]=\frac{1}{\lambda}
\]

方差：

\[
\text{Var}(T)=\frac{1}{\lambda^2}
\]

**记忆：**

- 泊松分布：单位时间内发生几次；
- 指数分布：下一次事件还要等多久；
- 泊松过程把这两个联系起来。

---

### 2.2 常见分布速记

| 分布 | 常见场景 | 关键性质 |
|---|---|---|
| 伯努利分布 | 一次 0/1 实验 | 参数 \(p\) |
| 二项分布 | \(n\) 次独立伯努利实验成功次数 | \(E=np\)，\(\text{Var}=np(1-p)\) |
| 泊松分布 | 单位时间内稀有事件发生次数 | \(E=\lambda\)，\(\text{Var}=\lambda\) |
| 指数分布 | 泊松过程的等待时间 | 无记忆性 |
| 均匀分布 | 区间内等可能 | 密度恒定 |
| 正态分布 | 大量独立因素叠加 | 中心极限定理 |

---

### 2.3 离散与连续概率

#### 离散型随机变量

离散型随机变量对应的是概率质量函数 PMF：

\[
P(X=x_i)=p_i
\]

期望：

\[
E[X]=\sum_i x_i p_i
\]

#### 连续型随机变量

连续型随机变量对应的是概率密度函数 PDF：

\[
f(x)
\]

期望：

\[
E[X]=\int x f(x)\,dx
\]

**一句话：**

- 离散：求和；
- 连续：积分。

---

### 2.4 最大似然估计 MLE

最大似然估计关心：

> 在参数 \(\theta\) 取什么值时，观测到当前这批数据的概率最大？

\[
\hat{\theta}_{MLE}=\arg\max_\theta P(D|\theta)
\]

通常等价于最大化 log-likelihood：

\[
\hat{\theta}_{MLE}=\arg\max_\theta \log P(D|\theta)
\]

**直觉：**  
数据已经发生了，反过来问“哪个参数最可能生成这些数据”。

---

### 2.5 最大后验估计 MAP

最大后验估计在 MLE 基础上加入先验：

\[
\hat{\theta}_{MAP}=\arg\max_\theta P(\theta|D)
\]

根据贝叶斯公式：

\[
P(\theta|D)\propto P(D|\theta)P(\theta)
\]

所以：

\[
\hat{\theta}_{MAP}=\arg\max_\theta P(D|\theta)P(\theta)
\]

**对比：**

| 方法 | 是否使用先验 | 优化目标 |
|---|---|---|
| MLE | 不使用 | \(P(D|\theta)\) |
| MAP | 使用 | \(P(D|\theta)P(\theta)\) |

**一句话：**

- MLE：只相信数据；
- MAP：数据 + 先验一起看。

---

### 2.6 p 值、显著性水平与置信区间

#### p 值

p-value 的含义是：

> 如果原假设 \(H_0\) 为真，观察到当前这么极端或更极端结果的概率。

不是“原假设为真的概率”。

#### 显著性水平 \(\alpha\)

显著性水平是人为设定的容忍犯错阈值，常见为：

\[
\alpha=0.05
\]

若：

\[
p<\alpha
\]

通常称结果显著。

#### 置信区间 CI

置信区间用于估计真实参数可能落在哪个范围。

例如 A/B Test 中，若估计点击率差异：

\[
p_A-p_B
\]

如果置信区间包含 0，例如：

\[
[-0.002,0.012]
\]

说明真实差异可能为 0，因此通常无法认为差异显著。

如果区间不包含 0，例如：

\[
[0.005,0.015]
\]

则说明 A 相比 B 的提升更有统计证据。

---

### 2.7 PDF 与 CDF

概率密度函数 PDF 和累积分布函数 CDF 是连续型随机变量的两个核心描述方式。

#### PDF：概率密度函数

PDF 通常记为：

\[
f(x)
\]

它描述的是随机变量在某个位置附近的“密度高低”。

**关键误区：**

连续型随机变量在单点处的概率为 0：

\[
P(X=x)=0
\]

所以 \(f(x)\) 不是 \(X=x\) 的概率。  
真正的概率来自区间面积：

\[
P(a<X\le b)=\int_a^b f(x)\,dx
\]

PDF 的基本性质：

\[
f(x)\ge 0
\]

\[
\int_{-\infty}^{+\infty}f(x)\,dx=1
\]

**注意：**  
PDF 的函数值可以大于 1，只要总面积为 1 即可。

---

#### CDF：累积分布函数

CDF 通常记为：

\[
F(x)
\]

定义为：

\[
F(x)=P(X\le x)
\]

它表示随机变量落在 \(x\) 左侧的累计概率。

CDF 的基本性质：

1. 取值范围在 \([0,1]\)；
2. 单调不减；
3. \[
   F(-\infty)=0,\quad F(+\infty)=1
   \]
4. 区间概率可由 CDF 相减得到：

\[
P(a<X\le b)=F(b)-F(a)
\]

---

#### PDF 与 CDF 的关系

PDF 积分得到 CDF：

\[
F(x)=\int_{-\infty}^{x}f(t)\,dt
\]

CDF 求导得到 PDF：

\[
f(x)=F'(x)
\]

前提是 \(F(x)\) 在该点可导。

**记忆：**

- PDF 像“密度图 / 瞬时速度”；
- CDF 像“累计进度条 / 已走路程”；
- PDF 积起来是 CDF；
- CDF 的斜率是 PDF。

---

#### 均匀分布例子

若：

\[
X\sim U(0,10)
\]

则 PDF 为：

\[
f(x)=
\begin{cases}
0.1, & 0\le x\le 10\\
0, & \text{其他}
\end{cases}
\]

CDF 为：

\[
F(x)=
\begin{cases}
0, & x<0\\
0.1x, & 0\le x\le 10\\
1, & x>10
\end{cases}
\]

例如：

\[
F(5)=0.5
\]

表示随机变量落在 5 及其左侧的概率是 50%。

---

#### PDF / CDF 考场速判

| 说法 | 判断 |
|---|---|
| PDF 是概率 | 错，PDF 是密度 |
| 连续变量单点概率非零 | 错，单点概率为 0 |
| PDF 可以大于 1 | 对 |
| CDF 可以下降 | 错，CDF 单调不减 |
| CDF 取值一定在 \([0,1]\) | 对 |
| 区间概率是 \(F(b)-F(a)\) | 对 |


## 3. 相关系数与回归指标

### 3.1 皮尔逊相关系数

皮尔逊相关系数衡量两个变量之间的线性相关性：

\[
r=\frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
\]

取值范围：

\[
[-1,1]
\]

| 取值 | 含义 |
|---|---|
| 接近 1 | 强正线性相关 |
| 接近 -1 | 强负线性相关 |
| 接近 0 | 线性相关弱 |

### 3.2 斯皮尔曼等级相关系数

Spearman 相关系数关注的是排序关系，而不是具体数值大小。

如果两个变量排序几乎完全一致，则 Spearman 接近 1。

**考试判断：**

- Pearson：看线性关系；
- Spearman：看单调排序关系。

---

### 3.3 SSE / SSR / SST 与 \(R^2\)

总平方和：

\[
SST=\sum_i (y_i-\bar{y})^2
\]

回归平方和：

\[
SSR=\sum_i (\hat{y}_i-\bar{y})^2
\]

残差平方和：

\[
SSE=\sum_i (y_i-\hat{y}_i)^2
\]

在普通最小二乘回归并含截距项时：

\[
SST=SSR+SSE
\]

决定系数：

\[
R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}
\]

**直觉：**

- SST：命运的总波动；
- SSR：模型解释掉的波动；
- SSE：模型无能为力的残差；
- \(R^2\)：模型在总波动里抢到了多少“功劳”。

---

### 3.4 RMSE

RMSE 是 Root Mean Squared Error：

\[
RMSE=\sqrt{\frac{1}{n}\sum_i (y_i-\hat{y}_i)^2}
\]

**名字倒过来读：**

1. Error：先算误差；
2. Squared：误差平方；
3. Mean：求平均；
4. Root：开根号。

---

## 4. 聚类算法

### 4.1 GMM：高斯混合模型

GMM 假设数据由多个高斯分布混合生成：

\[
p(x)=\sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)
\]

其中：

- \(\pi_k\)：第 \(k\) 个高斯成分的权重；
- \(\mu_k\)：均值；
- \(\Sigma_k\)：协方差矩阵。

GMM 常用 EM 算法估计参数。

#### EM 算法直觉

EM 分两步：

1. E-step：估计每个样本属于每个高斯成分的概率；
2. M-step：根据这些软分配重新估计参数。

**一句话：**  
GMM 是“反推数据可能由哪些高斯分布混合生成”。

---

### 4.2 K-means

K-means 通过最小化簇内平方误差进行聚类：

\[
\sum_{i=1}^n \|x_i-\mu_{c_i}\|^2
\]

特点：

- 需要提前指定 \(K\)；
- 倾向发现球形簇；
- 对初始中心敏感；
- 对异常值敏感；
- 是硬聚类。

---

### 4.3 DBSCAN

DBSCAN 是基于密度的聚类算法。

核心概念：

- \(\epsilon\)：邻域半径；
- MinPts：成为核心点所需的邻域点数；
- 核心点；
- 边界点；
- 噪声点。

特点：

- 能发现任意形状簇；
- 能识别噪声；
- 不需要提前指定簇数；
- 对参数 \(\epsilon\) 和 MinPts 敏感；
- 对密度差异很大的数据不一定好。

**一句话：**  
DBSCAN 靠“密度可达性”定义簇。

---

### 4.4 层次聚类 HAC

层次聚类常见为自底向上的凝聚式聚类：

1. 每个样本先各自成簇；
2. 每次合并距离最近的两个簇；
3. 直到满足停止条件。

常见簇间距离定义：

| 方法 | 定义 |
|---|---|
| Single Linkage | 两簇中最近样本对的距离 |
| Complete Linkage | 两簇中最远样本对的距离 |
| Average Linkage | 两簇样本对平均距离 |
| Ward Linkage | 合并后簇内平方误差增加最小 |

**Complete Linkage：**  
HAC 中经典的簇间距离定义，取两个簇之间最远点对距离。

---

### 4.5 谱聚类

谱聚类基于图和相似度矩阵：

1. 构造样本相似度图；
2. 构造图拉普拉斯矩阵；
3. 对拉普拉斯矩阵做特征分解；
4. 在低维特征空间中再用 K-means 聚类。

特点：

- 能处理非凸簇；
- 依赖相似度矩阵；
- 计算量较大；
- 对图构造方式敏感。

---

## 5. 线性代数与矩阵分析

### 5.1 线性子空间

线性子空间必须满足：

1. 对向量加法封闭；
2. 对标量乘法封闭；
3. 必须包含零向量。

为什么一定包含零向量？

如果 \(v\) 在子空间中，则对任意实数 \(k\)，都要求：

\[
kv
\]

仍在子空间中。取：

\[
k=0
\]

则：

\[
0\cdot v=0
\]

所以零向量必须在子空间中。

**一句话：**  
不含零向量的集合不可能是线性子空间。

---

### 5.2 二次型

二次型通常写成：

\[
x^T A x
\]

其中 \(A\) 通常取为对称矩阵。

根据 \(x^T A x\) 的符号，可判断：

| 条件 | 类型 |
|---|---|
| 对所有 \(x\neq 0\)，\(x^TAx>0\) | 正定 |
| 对所有 \(x\)，\(x^TAx\ge 0\) | 半正定 |
| 对所有 \(x\neq 0\)，\(x^TAx<0\) | 负定 |
| 有正有负 | 不定 |

---

### 5.3 正交矩阵

正交矩阵 \(Q\) 满足：

\[
Q^TQ=QQ^T=I
\]

因此：

\[
Q^{-1}=Q^T
\]

**记忆：**  
正交矩阵的转置就是逆矩阵。

---

### 5.4 奇异值分解 SVD

任意矩阵 \(A\in \mathbb{R}^{m\times n}\) 可以分解为：

\[
A=U\Sigma V^T
\]

其中：

- \(U\)：正交矩阵；
- \(V\)：正交矩阵；
- \(\Sigma\)：对角矩阵，主对角线上是奇异值；
- 奇异值通常按从大到小排列。

奇异值满足：

\[
\sigma_i=\sqrt{\lambda_i(A^TA)}
\]

其中 \(\lambda_i(A^TA)\) 是 \(A^TA\) 的特征值。

矩阵的秩等于非零奇异值的个数：

\[
rank(A)=\#\{\sigma_i>0\}
\]

**考试速记：**

- SVD：\(A=U\Sigma V^T\)；
- \(U,V\)：正交矩阵；
- \(\Sigma\)：对角非负，通常降序；
- rank = 非零奇异值个数。

---

### 5.5 线性变换矩阵的求法

若给定一组基：

\[
\{1,x,x^2\}
\]

以及线性变换 \(T\)，要求 \(T\) 在该基下的矩阵表示：

1. 分别把基向量 \(1,x,x^2\) 代入变换规则；
2. 把结果重新用这组基表示；
3. 得到每个结果的坐标列向量；
4. 把这些列向量从左到右拼起来。

**一句话：**  
“基向量进去，坐标列向量出来，列向量并排就是矩阵。”

---

### 5.6 秩、方程组与正规方程

设：

\[
A\in \mathbb{R}^{m\times n},\quad m\ge n
\]

若：

\[
rank(A)=n
\]

则说明 \(A\) 是**列满秩矩阵**。

---

#### 列满秩的核心等价性质

当 \(A\) 列满秩时，以下结论等价：

\[
rank(A)=n
\]

\[
\Longleftrightarrow A \text{ 的列向量线性无关}
\]

\[
\Longleftrightarrow A^TA \text{ 可逆}
\]

\[
\Longleftrightarrow \det(A^TA)>0
\]

\[
\Longleftrightarrow A^TA \text{ 正定}
\]

**考试速记：**

> 高瘦矩阵 + 列满秩  
> \(\Rightarrow\) 列向量线性无关  
> \(\Rightarrow\) \(A^TA\) 可逆且正定  
> \(\Rightarrow\) 正规方程有唯一解。

---

#### 为什么 \(A^TA\) 正定？

因为对任意 \(x\neq 0\)：

\[
x^TA^TAx=\|Ax\|^2
\]

如果 \(A\) 列满秩，则：

\[
x\neq 0 \Rightarrow Ax\neq 0
\]

所以：

\[
\|Ax\|^2>0
\]

因此：

\[
A^TA \succ 0
\]

也就是说 \(A^TA\) 是正定矩阵，自然可逆。

---

#### 正规方程一定有唯一解

最小二乘问题：

\[
\min_x \|Ax-b\|^2
\]

对应正规方程：

\[
A^TAx=A^Tb
\]

如果 \(A\) 列满秩，则 \(A^TA\) 可逆，因此：

\[
x=(A^TA)^{-1}A^Tb
\]

所以正规方程有唯一解。

**注意：**

即使原方程：

\[
Ax=b
\]

本身没有精确解，正规方程仍然可能有唯一解。  
这是最小二乘法的核心：原方程解不了，就找一个让残差平方和最小的 \(x\)。

---

#### 超定方程组不一定无解

当：

\[
m>n
\]

方程数量多于未知数数量，称为超定方程组。

一般随机情况下，超定方程组可能无解。  
但不能说它一定无解。

只要：

\[
b\in Col(A)
\]

也就是 \(b\) 落在 \(A\) 的列空间中，方程：

\[
Ax=b
\]

就有解。

如果 \(A\) 还列满秩，那么一旦有解，这个解就是唯一的。

---

#### 方程组解的黄金判定

对线性方程组：

\[
Ax=b
\]

其中 \(A\) 有 \(n\) 列。

第一步：看是否有解。

\[
rank(A)=rank([A|b])
\]

则有解。

\[
rank(A)\neq rank([A|b])
\]

则无解。

第二步：在有解的前提下，看解的个数。

| 条件 | 结论 |
|---|---|
| \(rank(A)=rank([A|b])=n\) | 唯一解 |
| \(rank(A)=rank([A|b])<n\) | 无穷多解 |
| \(rank(A)\neq rank([A|b])\) | 无解 |

**直觉：**

- 增广矩阵秩变大：\(b\) 带来了冲突，无解；
- 有解且满列秩：未知数全部被约束住，唯一解；
- 有解但不满列秩：存在自由变量，无穷多解。

---

#### 这类选择题的秒杀口诀

看到：

\[
A\in \mathbb{R}^{m\times n},\quad m\ge n,\quad rank(A)=n
\]

立刻写：

\[
A^TA \text{ 可逆}
\]

\[
A^TAx=A^Tb \text{ 有唯一解}
\]

\[
A \text{ 的列向量线性无关}
\]

\[
\det(A^TA)>0
\]

不要轻易选：

\[
Ax=b \text{ 一定无解}
\]

因为只要 \(b\in Col(A)\)，它就有解。


## 6. 数值计算

### 6.1 浮点误差：相近数相减

两个非常接近的数相减，会导致有效位数严重丢失。

这叫 catastrophic cancellation，灾难性消去。

例如：

\[
1.0000001-1.0000000
\]

结果很小，但前面很多有效数字被抵消掉。

**考试判断：**  
看到“两个相近大数相减”，优先想到有效位数丢失。

---

### 6.2 累加顺序

数值计算中，通常建议：

> 先累加小项，再累加大项。

也就是自小到大累加，可以减少小数被大数吞掉的误差。

---

### 6.3 截断误差

泰勒展开或数值近似中，被省略的第一项通常决定截断误差的主阶。

**记忆：**

> 余项的最低阶是主导项；往后多看一眼，看第一个被砍掉的项。

---

### 6.4 牛顿法

求方程：

\[
f(x)=0
\]

牛顿迭代公式：

\[
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
\]

特点：

- 局部收敛很快；
- 初值选得不好可能不收敛；
- 要求导数不为 0；
- 对重根可能收敛变慢。

---

### 6.5 不动点迭代收敛性

若迭代形式为：

\[
x_{k+1}=g(x_k)
\]

在不动点 \(x^\*\) 附近，如果：

\[
|g'(x^\*)|<1
\]

则通常局部收敛。

如果在某个区间内：

\[
|g'(x)|\le L<1
\]

则可用压缩映射直觉判断收敛。

**选择题快速法：**

- 看到 \(x_{k+1}=g(x_k)\)，看 \(|g'(x)|\)；
- 小于 1：倾向收敛；
- 大于 1：倾向发散；
- 等于 1：需要更细分析。

---

### 6.6 雅可比 / 高斯-赛德尔迭代收敛

解线性方程组：

\[
Ax=b
\]

迭代法收敛通常看迭代矩阵的谱半径：

\[
\rho(B)<1
\]

则迭代收敛。

#### 快速判断技巧

以下条件通常能保证收敛：

1. \(A\) 严格对角占优；
2. \(A\) 对称正定时，高斯-赛德尔收敛；
3. 小矩阵题可以直接构造迭代矩阵并看特征值模是否小于 1。

---

## 7. 机器学习基础模型

### 7.1 线性回归

线性回归目标通常是最小化平方误差：

\[
\min_w \sum_i (y_i-w^Tx_i)^2
\]

等价于最小化 SSE。

### 7.2 Huber 损失

Huber 损失在误差较小时像 MSE，在误差较大时像 MAE：

\[
L_\delta(a)=
\begin{cases}
\frac{1}{2}a^2, & |a|\le \delta\\
\delta(|a|-\frac{1}{2}\delta), & |a|>\delta
\end{cases}
\]

特点：

- 小误差：二次函数，平滑可导；
- 大误差：线性增长，降低异常值影响。

**一句话：**  
Huber = MSE 的平滑 + MAE 的抗异常值。

---

### 7.3 Batch Normalization

BN 通常对一个 mini-batch 的激活做归一化：

\[
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
\]

再通过可学习参数恢复表达能力：

\[
y=\gamma \hat{x}+\beta
\]

作用：

- 稳定中间激活分布；
- 缓解训练不稳定；
- 允许更大学习率；
- 带来一定正则化效果。

**考试判断：**

- BN 不会因为“稳定分布”而更容易过拟合；
- BN 通常反而有轻微正则化效果；
- BN 对 batch size 较敏感。

---

### 7.4 Dropout 与 BN 的顺序

Dropout 训练时会随机将部分神经元置零。  
如果 Dropout 放在 BN 前面，BN 统计到的是被 Dropout 扰动后的均值和方差。

推理时 Dropout 关闭，激活分布发生变化，但 BN 使用训练时的滑动统计量，可能导致 train/test mismatch。

**实践经验：**

- 同一个连续 block 中通常不强行同时使用 Dropout 和 BN；
- 若一定要用，常见做法是 Dropout 放在 BN 后面；
- Transformer 中更常见 LayerNorm + Dropout；
- CNN / ResNet 中 BN 已经常用，Dropout 使用相对谨慎。

---

### 7.5 LayerNorm

LayerNorm 对单个样本的特征维度做归一化，而不是跨 batch 统计。

公式：

\[
\mu=\frac{1}{H}\sum_{i=1}^{H}x_i
\]

\[
\sigma^2=\frac{1}{H}\sum_{i=1}^{H}(x_i-\mu)^2
\]

\[
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

特点：

- 不依赖 batch size；
- 常用于 Transformer；
- 对序列建模友好；
- 计算量通常是 \(O(H)\)，和特征维度成线性关系。

**BN vs LN：**

| 方法 | 统计维度 | 是否依赖 batch | 常见场景 |
|---|---|---|---|
| BN | batch 维度 | 依赖 | CNN |
| LN | 单样本特征维度 | 不依赖 | Transformer / NLP |

---

### 7.6 卷积核梯度累加

卷积操作中，同一个卷积核会在不同空间位置复用。

因此反向传播时，这个卷积核在不同位置产生的梯度贡献需要累加。

**一句话：**  
卷积核共享参数，所以梯度要把所有位置的贡献加起来。

---

## 8. 优化算法

### 8.1 SGD 在狭长山谷中的问题

如果损失曲面像一个狭长山谷：

- 曲率大的方向：SGD 容易左右震荡；
- 曲率小的方向：真正通向极小值，但推进很慢。

这就是病态曲率导致的优化困难。

---

### 8.2 Momentum

Momentum 会维护历史梯度方向的指数滑动平均：

\[
v_t=\beta v_{t-1}+\eta g_t
\]

\[
w_{t+1}=w_t-v_t
\]

直觉：

- 来回震荡的方向会互相抵消；
- 一直一致的方向会被累积加强。

**记忆：**  
Momentum 像推土机，压平横向震荡，沿着山谷方向蓄力前进。

---

### 8.3 Adam

Adam 同时维护：

- 一阶矩估计：梯度均值；
- 二阶矩估计：梯度平方均值。

公式直觉：

\[
m_t \approx E[g_t]
\]

\[
v_t \approx E[g_t^2]
\]

Adam 根据不同参数历史梯度大小，自适应调整学习率。

**注意：**  
Adam 不是严格在计算 Hessian 对角线，而是利用梯度平方的一阶矩作为自适应缩放依据。

---

## 9. NLP 与信息检索

### 9.1 分词粒度

看到以下关键词，直接归类：

| 方法 | 类型 |
|---|---|
| Whitespace / 正则分词 | 词级 Word-level |
| Character | 字符级 Character-level |
| BPE | 子词级 Subword-level |
| WordPiece | 子词级 Subword-level |
| SentencePiece | 子词级 Subword-level |

**一句话：**  
BPE / WordPiece / SentencePiece → 无脑子词级。

---

### 9.2 TF-IDF

TF-IDF = Term Frequency - Inverse Document Frequency。

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

其中：

\[
IDF(t)=\log \frac{N}{df(t)}
\]

直觉：

- 一个词在当前文档里出现多：重要；
- 但如果它在所有文档里都常见：不重要。

常用于：

- 关键词提取；
- 文档检索；
- 文本特征表示。

---

### 9.3 TextRank

TextRank 是基于图排序的关键词提取方法，思想类似 PageRank。

基本做法：

1. 把词看成图上的节点；
2. 共现关系作为边；
3. 迭代计算每个词的重要性分数。

适合：

- 单篇文档关键词提取；
- 无监督摘要 / 关键词抽取。

---

### 9.4 LDA：潜在狄利克雷分配

LDA 是主题模型。

它假设：

- 文档由多个主题混合生成；
- 每个主题由多个词构成概率分布。

输出通常是：

- 每篇文档的主题分布；
- 每个主题下的高概率词。

适合：

- 大型语料主题挖掘；
- 文档聚类；
- 主题分析。

**注意：**  
LDA 产出的是主题词，不是专门针对单篇文章的关键词提取算法。

---

### 9.5 最大熵模型 MaxEnt

最大熵模型的核心思想：

> 在满足已知约束的前提下，选择熵最大的分布。

直觉：

> 在不确定中保持最公平的盲猜。

常用于：

- 分类；
- 序列标注；
- 传统 NLP 特征模型。

---

### 9.6 TF-IDF / TextRank / LDA / MaxEnt 对比

| 方法 | 类型 | 常见用途 |
|---|---|---|
| TF-IDF | 统计权重 | 关键词提取、检索 |
| TextRank | 图排序 | 单篇关键词提取、摘要 |
| LDA | 主题模型 | 主题挖掘、大语料分析 |
| MaxEnt | 监督概率模型 | 分类、序列标注 |

**考试速判：**

- 单篇关键词：TF-IDF / TextRank；
- 大语料主题挖掘：LDA；
- 监督分类 / 序列标注：最大熵模型。

---

## 10. 深度学习与大模型

### 10.1 GQA / MQA 与 KV Cache

在 Transformer 推理中，KV cache 用于缓存历史 token 的 Key 和 Value，避免每步重复计算历史 token。

#### MHA

每个 attention head 都有自己的 K/V。

#### MQA

多个 Query heads 共享一组 K/V。

#### GQA

多个 Query heads 分组共享 K/V。

**效果：**

- 减少 KV cache 显存占用；
- 降低推理时 KV 读写带宽压力；
- 对 decoder 逐 token 生成尤其有用；
- 主要优化推理效率和显存，不是训练计算量的根本优化。

**考试一句话：**  
GQA / MQA 的核心收益是减少 KV cache 和推理带宽压力。

---

### 10.2 Beam Search

Beam Search 在生成时保留多个候选路径。

beam size 越大：

- 候选序列越多；
- 计算量越大；
- 显存/缓存开销可能增加；
- 结果通常更稳定，但可能更保守。

**一句话：**  
Beam Search 会显著增加计算量，因为它同时维护多条生成路径。

---

### 10.3 CTR

CTR = Click-Through Rate，点击率。

\[
CTR=\frac{\text{点击次数}}{\text{曝光次数}}
\]

常见于：

- 推荐系统；
- 广告排序；
- 搜索排序。

---

## 11. 分布式训练

### 11.1 Tensor Parallelism：张量并行

张量并行 TP 是在同一层内部，把矩阵乘法或权重矩阵按维度切到不同 GPU 上。

特点：

- 同一层权重被拆分；
- 多卡共同完成一层计算；
- 常用于大模型单层太大放不下时；
- 需要频繁通信。

**一句话：**  
TP 是“同一层内部切矩阵”。

---

### 11.2 ZeRO

ZeRO 主要用于拆分训练状态，降低显存占用。

可拆分：

1. 优化器状态；
2. 梯度；
3. 模型参数。

典型阶段：

| 阶段 | 拆分内容 |
|---|---|
| ZeRO-1 | 优化器状态 |
| ZeRO-2 | 优化器状态 + 梯度 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 |

**一句话：**  
ZeRO 是拆训练状态，尤其优化器状态、梯度和参数。

---

### 11.3 FSDP

FSDP = Fully Sharded Data Parallel。

它将模型参数、梯度、优化器状态在数据并行维度上切分。

常见流程：

- 前向前 all-gather 当前层参数；
- 计算后释放不需要的完整参数；
- 反向时 reduce-scatter 梯度；
- 优化器状态分片保存。

**一句话：**  
FSDP 是 PyTorch 体系里常见的全分片数据并行实现。

---

### 11.4 All-Reduce 与 Reduce-Scatter + All-Gather

All-Reduce 可以理解为：

> 每张卡都有一份数据，大家把数据求和/平均后，每张卡都拿到完整结果。

Reduce-Scatter + All-Gather 是 All-Reduce 的一种高效实现方式：

1. Reduce-Scatter：先把归约结果分片，每张卡拿一部分；
2. All-Gather：再把各分片收集起来，让每张卡都有完整结果。

这常用于大规模分布式训练中的梯度同步。

**和 FSDP / DeepSpeed 的层级关系：**

- Reduce-Scatter / All-Gather：通信原语；
- DDP / FSDP / DeepSpeed：训练并行框架或策略；
- FSDP / ZeRO 会大量使用这些通信原语。

---

### 11.5 Pipeline Bubble

流水线并行中，设备之间存在等待空闲时间，这部分就叫 pipeline bubble。

直觉：

> 前面的 stage 还没把活传过来，后面的 GPU 就只能等。

减少 bubble 的常见方式：

- 增加 micro-batch 数量；
- 使用更合理的流水线调度；
- 平衡各 stage 计算量。

---

## 12. 经典考试判断题速记

### 12.1 正定 / 半正定

- \(A^TA\)：一定半正定；
- \(A^TA\) 且 \(A\) 满列秩：正定且可逆；
- 高瘦矩阵列满秩：正规方程 \(A^TAx=A^Tb\) 有唯一解；
- 正定矩阵必须对称；
- \(A+A^T\)：一定对称，但不一定正定；
- 顺序主子式全大于 0：对称矩阵正定。

---

### 12.2 归一化

- BN：跨 batch 统计，常用于 CNN；
- LN：单样本内部特征统计，常用于 Transformer；
- BN 有一定正则化效果；
- Dropout 放在 BN 前可能导致统计分布不匹配；
- 小 batch 下 BN 可能不稳定。

---

### 12.3 数值计算

- 相近数相减：有效位数丢失；
- 小数加大数：小数可能被吞；
- 累加：通常先小后大；
- 截断误差：看第一个被省略项；
- 牛顿法：函数值除以导数；
- 不动点迭代：看 \(|g'(x)|<1\)。

---

### 12.4 聚类

| 算法 | 核心特点 |
|---|---|
| K-means | 球形簇、指定 K、对异常值敏感 |
| GMM | 软聚类、概率模型、EM 估计 |
| DBSCAN | 密度聚类、任意形状、识别噪声 |
| HAC | 层次结构、距离定义重要 |
| 谱聚类 | 图方法、非凸簇、计算较重 |

---

### 12.5 NLP

- BPE / WordPiece / SentencePiece：子词级；
- Whitespace / 正则分词：词级；
- Character：字符级；
- TF-IDF：关键词 / 检索；
- TextRank：图排序关键词；
- LDA：主题模型；
- MaxEnt：监督分类 / 序列标注。

---

### 12.6 大模型与分布式

- GQA / MQA：减少 KV cache，优化推理；
- Beam Search：保留多路径，增加计算量；
- TP：同一层内部切矩阵；
- ZeRO：切优化器状态、梯度、参数；
- FSDP：全分片数据并行；
- Reduce-Scatter / All-Gather：通信原语；
- Pipeline Bubble：流水线空转等待。

---

## 13. 一页速背版

### 13.1 公式

\[
1\ \text{Byte}=8\ \text{Bits}
\]

\[
E[\text{Exp}(\lambda)]=\frac{1}{\lambda}
\]

\[
\text{Var}[\text{Exp}(\lambda)]=\frac{1}{\lambda^2}
\]

\[
R^2=1-\frac{SSE}{SST}
\]

\[
RMSE=\sqrt{\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2}
\]

\[
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
\]

\[
A=U\Sigma V^T
\]

\[
rank(A)=\#\{\sigma_i>0\}
\]

\[
CTR=\frac{\text{点击次数}}{\text{曝光次数}}
\]

---

### 13.2 一句话判断

- 泊松过程的等待时间：指数分布；
- PDF 是密度，CDF 是累计概率，区间概率用 CDF 相减；
- 指数分布均值 \(1/\lambda\)，方差 \(1/\lambda^2\)；
- \(A^TA\) 半正定，满列秩正定；
- 正交矩阵转置等于逆；
- SVD 的 \(U,V\) 正交，\(\Sigma\) 对角非负；
- 非零奇异值个数等于秩；
- BN 稳定激活分布，有轻微正则化；
- LN 不依赖 batch，常用于 Transformer；
- Loss Scaling 防 FP16 梯度下溢；
- DBSCAN 可识别噪声和任意形状簇；
- KL 散度不是严格距离；
- BPE / WordPiece / SentencePiece 是子词级；
- Beam Search 会增加计算量；
- ZeRO 是拆训练状态；
- TP 是同层切矩阵；
- Pipeline Bubble 是流水线等待空闲。

---

## 14. 高频陷阱题

### 陷阱 1：牛顿法公式写反

错误：

\[
x_{k+1}=x_k-\frac{f'(x_k)}{f(x_k)}
\]

正确：

\[
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}
\]

---

### 陷阱 2：\(A+A^T\) 一定正定？

不一定。

它一定对称，但正定需要满足：

\[
x^T(A+A^T)x>0,\quad \forall x\neq 0
\]

---

### 陷阱 3：KL 散度是距离？

不是。  
KL 不对称，也不满足三角不等式。

---

### 陷阱 4：Loss Scaling 是防 overflow？

主要不是。  
它主要用于防 FP16 小梯度 underflow。

---

### 陷阱 5：蒙特卡洛不适合高维？

错误。  
蒙特卡洛的优势之一就是高维情况下不容易像网格积分那样爆炸。

---

### 陷阱 6：BN 会让模型更容易过拟合？

通常不是。  
BN 反而常常有一定正则化效果。

---



### 陷阱 8：超定方程组一定无解？

不一定。  
当 \(m>n\) 时，\(Ax=b\) 是超定方程组，通常更容易无解，但只要 \(b\in Col(A)\)，它就有解。  
如果 \(A\) 列满秩，那么一旦有解，解还是唯一的。

---

### 陷阱 9：PDF 的值就是概率？

不是。  
连续型变量单点概率为 0，PDF 是密度。  
区间概率才由积分得到：

\[
P(a<X\le b)=\int_a^b f(x)\,dx=F(b)-F(a)
\]


### 陷阱 7：LDA 是单篇关键词提取？

不准确。  
LDA 更偏主题模型，适合语料级主题挖掘。

---

## 15. 最后复习建议

如果时间很紧，优先背下面几类：

1. **公式类**  
   牛顿法、RMSE、\(R^2\)、指数分布均值方差、CTR、SVD。

2. **判断类**  
   正定 / 半正定、列满秩 / 正规方程、PDF / CDF、BN / LN、Dropout + BN、KL 散度、Loss Scaling。

3. **分类类**  
   聚类算法特点、NLP 关键词提取方法、分词粒度、大模型并行策略。

4. **陷阱类**  
   牛顿法公式别写反；\(A+A^T\) 不一定正定；Loss Scaling 防下溢；KL 不是距离。

---

## 16. 考前 30 秒口播版

- Byte 是 8 bit，FP16 两字节，FP32 四字节。
- 泊松过程的等待时间服从指数分布，均值 \(1/\lambda\)，方差 \(1/\lambda^2\)。
- \(A^TA\) 一定半正定，满列秩才正定；\(A+A^T\) 只是对称。
- 正定看对称矩阵的顺序主子式全大于 0。
- SVD：\(A=U\Sigma V^T\)，\(U,V\) 正交，非零奇异值个数等于 rank。
- BN 看 batch，LN 看单样本特征；Transformer 常用 LN。
- Dropout 放 BN 前容易造成训练和推理分布不一致。
- Loss Scaling 防 FP16 梯度下溢。
- DBSCAN 能找任意形状簇和噪声。
- TF-IDF / TextRank 做关键词，LDA 做主题，MaxEnt 做监督分类。
- BPE / WordPiece / SentencePiece 都是子词级。
- Beam Search 多路径，计算量更大。
- TP 切矩阵，ZeRO 切优化器/梯度/参数，FSDP 是全分片数据并行。

