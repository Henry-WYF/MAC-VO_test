# 回环检测阶段A

阶段A仅实现严格因果的离线 ORB-BoW 地点候选检索，不生成位姿图回环边，也不会修改MAC-VO输出轨迹。

## 使用流程

1. 使用**独立训练序列**和 `MACVO_Performant_LoopPhaseA.yaml` 运行MAC-VO。若词典尚不存在，系统仍会在 `<实验结果目录>/loop_closure` 中缓存ORB描述子，但会跳过BoW查询。

2. 使用独立训练序列的缓存生成共享词典：

   ```bash
   python Scripts/LoopClosure/train_orb_vocabulary.py \
     <训练序列结果目录>/loop_closure \
     --output Model/ORB_BoW_4096.npz
   ```

3. 使用同一阶段A配置和生成的词典运行目标序列。实验结果目录中将包含：

   - `loop_closure/index.json`：有效回环缓存帧的索引；
   - `loop_closure/frames/*.pt`：可重建双目图像及深度数据的帧缓存；
   - `loop_closure/queries.json`：严格因果的top-10候选检索结果。

## 重要约束

- 词典及冻结IDF必须由独立训练序列生成，不得使用待评估序列训练。
- 运行期间只缓存帧、ORB描述子和BoW向量，不将帧提前加入查询数据库。
- 终止阶段按照 `sensor_frame_idx` 从小到大回放：先查询历史数据库，再插入当前帧。
- `queries.json` 中每个候选都必须满足：

  ```text
  candidate.sensor_frame_idx < current.sensor_frame_idx
  ```

- 回环缓存或检索发生故障时，默认只禁用回环模块，原MAC-VO仍继续运行并保存轨迹。

## 当前阶段不包含

- FlowFormerCov跨时刻几何验证；
- PnP位姿估计；
- GlobalPGO回环边接入；
- Huber鲁棒损失；
- 连续地点簇一致性；
- 在线回环线程。

这些功能将在阶段A候选召回通过验收后分阶段实现。
