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

## Phase B.5：ORB/Flow 并行观察

Phase B.5 默认是 `observe`，不修改 VO pose、GlobalPGO 或 `LoopConstraint.information`。服务器上先从同一缓存运行：

```bash
python3 Scripts/AdHoc/RunLoopPhaseBOffline.py \
  --result-dir RESULT_DIR \
  --output-dir RESULT_DIR/loop_phase_b5_observe \
  --config Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml \
  --device cuda \
  --phase-b5-mode observe
```

结果包含共享前缀的 `phase_b5_calibration_manifest.json`，以及相互隔离的 `orb_observe/`、`flow_all_bow_observe/`、`flow_orb_supported_observe/` 和 `forced_control_shadow/`。先在开发集冻结配置中的两个 absolute log-risk cap，再用可信 calibration manifest 重放，才能生成带 point cap 的 selector shadow A/B。

第一次输出只用于查看开发集 prefix 的 `pair_risk_median` 与 `pair_risk_q95`。在查看冻结集结果前写死两个 absolute cap，并对开发集和冻结集分别重放（两者使用相同 cap）：

```bash
python3 Scripts/AdHoc/RunLoopPhaseBOffline.py \
  --result-dir RESULT_DIR \
  --output-dir RESULT_DIR/loop_phase_b5_calibrated \
  --config Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml \
  --device cuda --phase-b5-mode observe \
  --phase-b5-absolute-median-cap FROZEN_MEDIAN_CAP \
  --phase-b5-absolute-q95-cap FROZEN_Q95_CAP
```

post-prefix 冻结评价命令为：

```bash
python3 Scripts/AdHoc/EvaluateLoopPhaseB5.py evaluate \
  --branch RESULT_DIR/loop_phase_b5_observe/flow_all_bow_observe/verification.json \
  --manifest RESULT_DIR/loop_phase_b5_observe/phase_b5_calibration_manifest.json \
  --labels RESULT_DIR/loop_labels_frozen.json \
  --legacy-verification RESULT_DIR/loop_phase_b5_observe/loop_verification.json \
  --ref-poses RESULT_DIR/ref_poses.npy \
  --cache-index RESULT_DIR/loop_closure/index.json \
  --record-dir RESULT_DIR/loop_closure \
  --role development \
  --output RESULT_DIR/loop_phase_b5_observe/evaluation.json
```

开发集和冻结集的 pair gate 都通过后，先生成只晋升 pair gate 的中间 manifest：

```bash
python3 Scripts/AdHoc/EvaluateLoopPhaseB5.py promote-pair \
  --manifest DATASET_CALIBRATION_MANIFEST \
  --development-eval DEVELOPMENT_EVAL \
  --frozen-eval FROZEN_EVAL \
  --population all_bow_candidates \
  --output DATASET_PAIR_PROMOTED_MANIFEST
```

用中间 manifest 再运行一次 `observe`，此时 selector shadow 使用“通过 pair gate 的 calibration 帧对”得到的 Q95 point cap。两套 selector 评价也通过后，再运行 `promote` 生成最终 manifest。`apply` 只接受同时记录开发集和冻结集通过、且明确晋升 Flow pair gate 与 selector 的最终 manifest。完整 VO 与 GlobalPGO 仍不属于 Phase B.5 验收步骤。

开发集和冻结集有各自的缓存摘要与 calibration manifest，因此 `promote-pair` 和最终 `promote` 都应针对两个数据集各执行一次；两次命令共享同一对 development/frozen evaluation 文件，但 `--manifest` 指向当前数据集自己的 manifest。中间 manifest 的 selector 重放示例：

```bash
python3 Scripts/AdHoc/RunLoopPhaseBOffline.py \
  --result-dir RESULT_DIR \
  --output-dir RESULT_DIR/loop_phase_b5_selector_shadow \
  --config Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml \
  --device cuda --phase-b5-mode observe \
  --phase-b5-manifest DATASET_PAIR_PROMOTED_MANIFEST
```
