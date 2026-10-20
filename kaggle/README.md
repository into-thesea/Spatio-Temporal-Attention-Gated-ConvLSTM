# 在 Kaggle 上运行

训练和评测脚本与本地用法一致。在 Kaggle 上：

1. 将本仓库（或处理后的 `data/`）上传为 Dataset。
2. 复制到 `/kaggle/working`，因为 `/kaggle/input` 是只读的。
3. 使用 T4 GPU 运行训练阶段。

```bash
python run_kaggle.py --dataset metrla --stage baseline
python run_kaggle.py --dataset metrla --stage seeds
python run_kaggle.py --dataset pemsbay --stage baseline
```

`--stage baseline` 会为一个数据集训练三种配置：

- 线性时间编码
- 周期时间编码
- 周期时间编码 + 残差预测

权重写入 `models/`，评测 CSV 写入 `results/`。  
session 结束前请从 Kaggle 的 Output 页签下载。

## 说明

- `data/` 中需要有处理后的 `<dataset>_grid_*.npy` 文件；如果没有，先下载原始数据并运行
  `python scripts/prepare_data.py --dataset metrla`。
- 每次训练使用一个随机种子。需要均值和方差时，用不同 `--seed` 重复运行。
- 脚本默认 40 个 epoch，早停 patience 为 10。
