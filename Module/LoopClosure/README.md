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
- 运行期间只缓存帧和公共ORB描述子；各后端在回放时重算自己的BoW。
- 终止阶段按照 `sensor_frame_idx` 从小到大回放：先将已达到时间排除间隔的pending帧入库，再查询当前帧。
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

## DBoW2 Phase A 候选后端

`dbow2_orb` 使用固定 ORB-SLAM3 v1.0 DBoW2 子集和标准文本
`ORBvoc.txt`。VO 运行期间只缓存公共 OpenCV ORB描述子；custom 与 DBoW2
均在 `detect_all()` 或离线重放时从描述子重算自己的 BoW。

标准词典可通过 `bash Scripts/LoopClosure/download_orbvoc.sh Model` 获取；脚本会
同时打印压缩包和解压文本的 SHA-256。词典训练数据来源未获权威说明，实验元数据
固定记录为 `not_authoritatively_verified`。

服务器可保留原镜像并增量构建：

```bash
docker tag macvo:latest macvo:pre-dbow2
docker build --build-arg MACVO_BASE_IMAGE=macvo:pre-dbow2 \
  -t macvo:phase-a-dbow2 -f Docker/Dockerfile.dbow2 .
```

首先运行完整词典/真实缓存构建 gate：

```bash
python3 Scripts/LoopClosure/run_dbow2_docker_gate.py \
  --vocabulary Model/ORBvoc.txt \
  --vocabulary-archive Model/ORBvoc.txt.tar.gz \
  --cache /path/to/result/loop_closure \
  --output /path/to/dbow2_gate
```

同一缓存的双后端 raw 重放：

```bash
python3 Scripts/AdHoc/RunLoopPhaseAOffline.py RESULT_DIR \
  --backend custom_binary --vocabulary Model/ORB_BoW_4096.npz \
  --output-dir RESULT_DIR/loop_phase_a_custom_raw

python3 Scripts/AdHoc/RunLoopPhaseAOffline.py RESULT_DIR \
  --backend dbow2_orb --vocabulary Model/ORBvoc.txt \
  --output-dir RESULT_DIR/loop_phase_a_dbow2_raw
```

随后用 `prepare_loop_review_set.py` 生成隐藏后端/排名/分数的审核图像，人工填写
`loop_review_items.json` 的 `label` 后，使用：

```bash
python3 Scripts/LoopClosure/prepare_loop_review_set.py \
  --cache RESULT_DIR/loop_closure --gt-poses RESULT_DIR/ref_poses.npy \
  --gt-index-space sensor_frame_idx \
  --custom-queries RESULT_DIR/loop_phase_a_custom_raw/queries.json \
  --dbow2-queries RESULT_DIR/loop_phase_a_dbow2_raw/queries.json \
  --output REVIEW_DIR

python3 Scripts/LoopClosure/evaluate_loop_retrieval.py freeze \
  --review REVIEW_DIR/loop_review_items.json \
  --output REVIEW_DIR/loop_labels.json

python3 Scripts/LoopClosure/evaluate_loop_retrieval.py evaluate \
  --labels REVIEW_DIR/loop_labels.json \
  --custom-queries RESULT_DIR/loop_phase_a_custom_raw/queries.json \
  --dbow2-queries RESULT_DIR/loop_phase_a_dbow2_raw/queries.json \
  --output REVIEW_DIR/development_metrics.json
```

`queries.json` schema 2 保存后端、词典摘要、eligible数据库规模以及分数过滤前后
计数；Phase B 离线入口同时接受 schema 1 和 2。
