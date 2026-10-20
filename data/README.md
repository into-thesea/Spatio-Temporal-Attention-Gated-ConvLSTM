# 数据说明

仓库不包含原始数据。

## METR-LA

- 207 个环形检测器，洛杉矶县
- 5 分钟粒度的速度记录
- 原始文件放置位置：
  - `data/metr-la.h5`
  - `data/graph_sensor_locations.csv`

## PeMS-BAY

- 325 个传感器，旧金山湾区
- 原始文件放置位置：
  - `data/pems-bay.h5`
  - `data/graph_sensor_locations_bay.csv`

原始数据集由各自的数据提供方发布，下载和再分发时请遵循其使用条款。

## 生成网格张量

```bash
python scripts/prepare_data.py --dataset metrla
python scripts/prepare_data.py --dataset pemsbay
```

输出：

```
<dataset>_grid_speed.npy
<dataset>_grid_observed.npy
<dataset>_grid_strict_mask.npy
<dataset>_grid_dilated_mask.npy
<dataset>_grid_meta.json
```

`observed` 标记真实的非零观测；`dilated_mask` 之外的格点不参与模型计算。
