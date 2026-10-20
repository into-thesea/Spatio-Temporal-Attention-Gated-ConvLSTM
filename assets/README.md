# 图表资源

由 `python scripts/make_figures.py` 和
`python scripts/make_model_details.py` 生成：

- `architecture.png`：模型总览
- `detail_stag.png`：STAG 空间门控
- `detail_convlstm.png`：ConvLSTM 单元与四个门
- `detail_attention.png`：时间注意力与融合
- `detail_output.png`：输出层与掩码损失
- `horizon_mae.png`：两个数据集的逐 horizon MAE
- `spatial_error_metrla.png`：METR-LA 的 60 分钟逐格 MAE
- `spatial_error_pemsbay.png`：PeMS-BAY 的 60 分钟逐格 MAE
- `prediction_example_metrla.png`：METR-LA 的预测 vs 真值
- `prediction_example_pemsbay.png`：PeMS-BAY 的预测 vs 真值
- `map_prediction_metrla.png`：METR-LA 的 15/30/60 分钟地图预测
- `map_prediction_pemsbay.png`：PeMS-BAY 的 15/30/60 分钟地图预测

空间误差图和预测示例需要权重与网格数据；horizon 曲线只依赖
`results/reference/` 中的 CSV。
