# STAGC：基于 ConvLSTM 与注意力门控的短时交通速度预测

STAGC 是一个轻量的时空预测模型，用于区域级短时交通速度预测。模型将稀疏的路网传感器映射到 16×16 网格，并以 5 分钟为步长预测未来 1 小时的速度场。

> 名称说明：STAGC 是本项目对 Spatio-Temporal Attention Gated ConvLSTM 的缩写，不是已发表的同名交通预测模型。公开文献中另有 STAGCN、ST-GACN、ST-GATC 等相近名称，请注意区分。

## 项目概述

短时速度预测可用于导航 ETA、车队调度和交通监测。交通网络本身是非欧几里得结构，图模型能够保留道路拓扑，但依赖预定义的邻接矩阵，计算成本会随节点规模增长。网格卷积避免了邻接矩阵，单格计算成本固定，但会损失一部分道路连通性。

STAGC 保留网格建模方式，并在每个 ConvLSTM 单元前加入 STAG 空间注意力门控。门控利用上一时刻的隐状态对当前输入进行重加权，解码器则通过时间注意力从编码器历史状态中检索相关信息。模型输入为 12 帧速度与时间特征，输出为未来 12 帧速度图，损失函数只在有有效观测的格点上计算掩码 Huber 损失。

模型定位为区域级 16×16 网格、1 小时预测窗口，不使用事故、天气、流量或排队长度等外部数据。

## 实验结果

测试集，仅统计有效传感器观测，12 步预测（5–60 分钟）：

| 数据集 | 配置 | 平均 MAE | 60min MAE | 60min RMSE | 60min MAPE |
|---|---|---:|---:|---:|---:|
| METR-LA | 线性时间编码 | 2.711 | 3.099 | 5.302 | 6.84% |
| METR-LA | 周期时间编码 | **2.678** | **3.028** | **5.209** | **6.70%** |
| METR-LA | 周期编码 + 残差预测 | 2.685 | 3.104 | 5.333 | 6.97% |
| PeMS-BAY | 线性时间编码 | 1.432 | 1.731 | 3.340 | 3.28% |
| PeMS-BAY | 周期时间编码 | 1.438 | **1.717** | 3.265 | 3.25% |
| PeMS-BAY | 周期编码 + 残差预测 | **1.386** | 1.723 | **3.258** | 3.26% |

逐 horizon 指标见 `results/reference/`。

16 核笔记本 CPU 上的推理 benchmark：

| 项目 | 数值 |
|---|---|
| 参数量 | 0.92M |
| PyTorch 权重 | 3.69 MB |
| ONNX 模型 | 4.24 MB |
| ONNX Runtime，batch=1 | 23.4 ms，42.7 samples/s |

延迟在笔记本 CPU 上测得，不代表 Jetson、树莓派等真实边缘设备。

## 模型结构

- 输入：12 帧 ×（速度 + 时间特征），16×16 网格
- 编码器：两层 ConvLSTM，每个单元前接 STAG 空间门控
- 解码器：12 步递归预测 + 时间注意力
- 损失：只在有效观测格点上计算的掩码 Huber 损失
- 输出：未来 12 帧速度图

时间特征支持两种形式：线性编码（时段、星期）和周期编码（sin/cos）。当前主模型使用周期编码。

## 门控机制与 STAG 的定位

STAG 不是凭空提出的结构。它和 LSTM、GRU、Highway、GLU、SE-Net 这一类“门控 + 加权”工作有共同思想：用一个可学习的 gate 控制信息保留、放大或抑制。区别主要在**作用维度**和**插入位置**。

### LSTM 与 ConvLSTM

LSTM 用输入门、遗忘门、输出门控制 cell state 的写入、保留和输出：

```text
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_g [h_{t-1}, x_t] + b_g)
h_t = o_t ⊙ tanh(c_t)
```

ConvLSTM 把全连接替换成卷积，使门控在 H×W 空间上共享，能够保留网格结构。本项目的 ConvLSTM 单元就是这一形式。

### GRU

GRU（Cho et al., 2014）简化了 LSTM，用更新门和重置门：

```text
z_t = sigmoid(W_z [h_{t-1}, x_t])
r_t = sigmoid(W_r [h_{t-1}, x_t])
h~_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
```

GRU 的 `(1-z)` 与 `z` 是一种时间维度的残差门控：保留旧隐状态，同时写入新候选状态。DCRNN 的循环部分使用了 GRU 结构。

### Highway Network

Highway Network 用 transform gate `T(x)` 在“变换后的特征”和“原始输入”之间做加权：

```text
y = H(x) ⊙ T(x) + x ⊙ (1 - T(x))
```

它和 STAG 的 `x ⊙ (1+g)` 更接近：都保留了原始输入路径，只是 gate 的位置和维度不同。

### GLU 与 SE-Net

GLU（Dauphin et al., 2017）用一路卷积特征去门控另一路：

```text
GLU(X) = (XW + b) ⊙ sigmoid(XV + c)
```

SE-Net（Hu et al., 2017）在通道维度做全局池化后生成通道权重：

```text
s = sigmoid(FC(GAP(X)))
X' = X ⊙ s
```

Graph WaveNet 的 temporal convolution 也使用了 gated convolution 的思路。

### STAG 的定位

本项目的 STAG：

```text
g_t = sigmoid(Conv7x7([x_t, h_{t-1}]))
x'_t = x_t ⊙ (1 + g_t)
```

它和上述机制的关系：

- 使用 Sigmoid gate + 逐元素乘法，和 GRU、GLU、SE-Net 一样；
- 保留原始输入 `x_t`，和 Highway、GRU 的残差思想一致；
- 但 gate 是一张 16×16 的空间图，不是时间隐状态，也不是通道向量；
- 它作用在 ConvLSTM 单元之前，用上一时刻隐状态 `h_{t-1}` 判断当前帧哪些空间位置更重要。

因此更准确的说法是：

> STAG 是一个空间维度的输入重校准门控，借鉴了 GRU/Highway 的“门控 + 残差”思想，但不是 GRU 的替代品。

项目中的 Sigmoid 版本满足 `1 + g_t ∈ (1,2)`，因此只会放大输入；仓库中也保留了一个 tanh 门控变体，使用 `x ⊙ (1 + tanh(...))`，使 gate 范围变为 `[0,2]`，可以放大也可以抑制。主模型使用 Sigmoid 版本。

### 对比表

| 机制 | 作用维度 | 位置 | Gate | 主要目的 |
|---|---|---|---|---|
| LSTM | 时间 / 隐状态 | 循环单元内部 | Sigmoid / Tanh | 长期记忆 |
| GRU | 时间 / 隐状态 | 循环单元内部 | Sigmoid / Tanh | 简化记忆更新 |
| Highway | 层间 | 前馈网络 | Sigmoid | 信息高速公路 |
| GLU | 特征 / 通道 | 卷积层 | Sigmoid | 特征选择 |
| SE-Net | 通道 | 卷积块 | Sigmoid | 通道重校准 |
| STAG | 空间 H×W | ConvLSTM 之前 | Sigmoid | 空间输入重校准 |

## 参考资料

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997
- Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*, 2014  
  https://arxiv.org/abs/1406.1078
- Chung et al., *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*, 2014  
  https://arxiv.org/abs/1412.3555
- Srivastava et al., *Training Very Deep Networks*, 2015  
  https://arxiv.org/abs/1507.06228
- Dauphin et al., *Language Modeling with Gated Convolutional Networks*, 2017  
  https://arxiv.org/abs/1612.08083
- Hu et al., *Squeeze-and-Excitation Networks*, 2017  
  https://arxiv.org/abs/1709.01507
- Shi et al., *Convolutional LSTM Network*, 2015
- Li et al., *Diffusion Convolutional Recurrent Neural Network*, 2018  
  https://arxiv.org/abs/1707.01926
- Wu et al., *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, 2019  
  https://arxiv.org/abs/1906.00121

## 图表说明

### 模型结构

总览图展示编码器-解码器路径、空间门控、时间注意力分支和融合层。

![模型结构](assets/architecture.png)

详细结构拆成四个模块，分别说明通道数、张量形状、门控函数、激活函数和残差路径。图中堆叠矩形表示带通道维度的特征图/张量，单层矩形表示卷积、激活、Softmax 等操作。

#### STAG 空间门控

输入 $x_t$ 与上一时刻隐状态 $h_{t-1}$ 拼接后，经过 7×7 卷积（69→1 通道）和 Sigmoid 得到空间门控 $g_t$。输出为 $x'_t = x_t \odot (1+g_t)$，虚线表示从 $x_t$ 到乘法节点的残差路径。

![STAG 空间门控](assets/detail_stag.png)

#### ConvLSTM 单元

拼接后的特征经过 3×3 卷积（69→256 通道），拆成输入门、遗忘门、输出门和候选状态四组，每组 64 通道。输入门、遗忘门、输出门使用 Sigmoid，候选状态使用 Tanh；随后按 $c_t=f_t\odot c_{t-1}+i_t\odot g_t$ 和 $h_t=o_t\odot\tanh(c_t)$ 更新状态。

![ConvLSTM 单元](assets/detail_convlstm.png)

#### 时间注意力与融合

解码器状态和编码器历史状态先做全局平均池化，再通过 Q/K 线性层（64→64）计算缩放点积，Softmax 得到 12 个历史时刻的权重；V 直接使用编码器状态，不做额外投影。上下文向量与解码器状态拼接后，经过 1×1 卷积（128→64）和 Dropout 得到融合特征。

![时间注意力与融合](assets/detail_attention.png)

#### 输出层与损失

融合特征经过 1×1 卷积（64→1）输出单通道速度图，输出层使用线性激活，不加 Sigmoid/Tanh；随后按训练集的最小值和最大值反归一化。损失函数为掩码 Huber 损失，只在有效观测格点上计算；预测帧会作为下一步解码器输入，递归 12 次。

![输出层与损失](assets/detail_output.png)

### 逐 horizon 精度

左图为 METR-LA，右图为 PeMS-BAY。每条曲线对应一种输入/目标配置：线性时间编码、周期时间编码、周期编码 + 残差预测。

![逐 horizon MAE](assets/horizon_mae.png)

### 地图预测结果

每张地图对比 15、30、60 分钟的预测值与真实值。点为传感器位置，按速度着色；上排为真实值，下排为模型预测。两个数据集使用不同的地理范围，因此分别展示。

![METR-LA 地图预测](assets/map_prediction_metrla.png)

![PeMS-BAY 地图预测](assets/map_prediction_pemsbay.png)

底图：© OpenStreetMap contributors, HOT。

### 误差分析

下图统计 METR-LA 测试集在 60 分钟处的逐格绝对误差。误差集中在少数格点，而不是均匀分布。

![METR-LA 空间 MAE](assets/spatial_error_metrla.png)

下图为 PeMS-BAY 的同一分析。颜色尺度按数据集单独设置，建议比较空间分布形态，而不是直接比较绝对色值。

![PeMS-BAY 空间 MAE](assets/spatial_error_pemsbay.png)

下面是 METR-LA 的一个测试窗口，展示 60 分钟预测值与真实值。

![METR-LA 预测示例](assets/prediction_example_metrla.png)

PeMS-BAY 的对应示例。

![PeMS-BAY 预测示例](assets/prediction_example_pemsbay.png)

图表由 `scripts/make_figures.py`、`scripts/make_model_details.py` 和 `scripts/make_map_figures.py` 生成。地图叠加图需要先安装 `requirements-figures.txt`。

## 仓库结构

```
data_core/data.py              数据加载、时间切分、指标
models/stagc.py                基础模型
train.py                       训练
eval.py                        评测
scripts/prepare_data.py        原始传感器数据 -> 网格张量
scripts/audit_data.py          网格数据审计
scripts/benchmark_infer.py     CPU/ONNX benchmark
scripts/make_figures.py        README 图表
scripts/make_model_details.py  详细模型图
scripts/make_map_figures.py    地图叠加图
assets/                        生成的图表
tests/                         数据与指标测试
results/reference/             参考指标与训练日志
kaggle/                        Kaggle 训练入口
```

## 环境安装

```bash
python -m pip install -r requirements.txt
```

## 数据准备

下载 METR-LA 和 PeMS-BAY，并将原始文件放到 `data/` 下。具体文件名见 `data/README.md`。

生成网格张量：

```bash
python scripts/prepare_data.py --dataset metrla
python scripts/prepare_data.py --dataset pemsbay
```

输出文件：

```
data/metrla_grid_speed.npy
data/metrla_grid_observed.npy
data/metrla_grid_strict_mask.npy
data/metrla_grid_dilated_mask.npy
data/metrla_grid_meta.json
```

## 训练

```bash
python train.py --dataset metrla --model base \
  --time-encoding cyclic --residual \
  --epochs 40 --batch-size 32 --seed 42 --patience 10
```

权重保存到 `models/`，日志保存到 `results/`。  
如果存在 `models/<tag>_last.pth`，训练会自动续训；使用 `--fresh` 可以强制从头训练。

## 评测

```bash
python eval.py --checkpoint models/base_cyclic_s42.pth --split test
```

## 测试

```bash
python -m unittest discover -s tests -v
```

部分测试在缺少权重或样本数据时会自动跳过。

## 项目说明与 AI 辅助

这是一个个人学习与作品集项目，重点在于完整走通数据准备、模型训练、评测和 CPU 推理的流程。

代码和文档的部分内容使用了 AI 辅助起草。所有报告数字均来自实际运行本仓库代码；已知局限列在下方以及 `MODEL_CARD.md` 中。作者对内容和实现验证负责。

## 局限

- 16×16 网格是区域级分辨率，不是车道级。
- 预测窗口只有 1 小时。
- 不使用事故、天气、流量或排队长度数据。

## 许可证

MIT，详见 `LICENSE`。
