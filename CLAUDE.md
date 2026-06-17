# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAC-VO is an ICRA 2025 Best Paper Award-winning learning-based **stereo visual odometry** system. Its core innovation is propagating **uncertainty (covariance)** from neural network outputs through the entire pipeline into pose graph optimization, so observations are automatically weighted by their predicted reliability.

## Common Commands

### Run MAC-VO on a sequence
```bash
python MACVO.py --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/TartanAir_example.yaml
# Fast mode (mixed precision, ~2x speed):
python MACVO.py --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/TartanAir_example.yaml
# With 3D visualization:
python MACVO.py --useRR --odom ... --data ...
```

### Evaluate and visualize results
```bash
python -m Evaluation.EvalSeq --spaces SPACE_PATH_0 [SPACE_PATH_1 ...]
python -m Evaluation.PlotSeq --spaces SPACE_PATH_0 [SPACE_PATH_1 ...]
```

### Run tests
```bash
pytest Scripts/UnitTest/
# Exclude tests requiring TensorRT:
pytest Scripts/UnitTest/ -m "not trt"
# Single test file:
pytest Scripts/UnitTest/test_config_macvo.py
```

### Static analysis (pyright)
```bash
pyright
```
Configuration is in `pyproject.toml` under `[tool.pyright]`. Third-party network modules (`Module/Network/PWCNet`, `FlowFormer`, etc.) are excluded.

### Run baselines
```bash
python -m Scripts.Experiment.Experiment_DPVO --odom Config/Experiment/Baseline/DPVO/DPVO.yaml
python -m Scripts.Experiment.Experiment_TartanVO --odom Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml
```

## Architecture: Modular Plugin System

The codebase uses a strict **Interface → Implementation** pattern. Interfaces are abstract base classes; concrete implementations are loaded **dynamically from YAML config** via `Interface.instantiate(type_string, args)`. This means you can swap any component without touching code.

### Core Interfaces (all under `Module/`)

| Interface | File | Role |
|---|---|---|
| `IFrontend` | `Module/Frontend/Frontend.py` | Joint depth + optical flow estimation with covariance |
| `IStereoDepth` | `Module/Frontend/StereoDepth.py` | Dense stereo depth + depth covariance |
| `IMatcher` | `Module/Frontend/Matching.py` | Dense optical flow + flow covariance (between left frames) |
| `IKeypointSelector` | `Module/KeypointSelector.py` | Selects sparse keypoints for tracking |
| `ICovariance2to3` | `Module/Covariance/Project2to3.py` | Projects 2D pixel uncertainty → 3×3 spatial covariance |
| `IMotionModel` | `Module/MotionModel.py` | Initial pose guess for each frame |
| `IObservationFilter` | `Module/OutlierFilter.py` | Rejects bad observations |
| `IKeyframeSelector` | `Module/KeyframeSelector.py` | Decides which frames are keyframes |
| `IOptimizer` | `Module/Optimization/Interface.py` | Pose graph optimization (sequential or parallel subprocess) |
| `IMapProcessor` | `Module/MapProcessor.py` | Post-processing: interpolation, smoothing |

### Configuration System

YAML configs support `!include <path>.yaml` and `!flatten` custom tags. Configs are split into three concerns:
- **Experiment configs** (`Config/Experiment/`) — which modules, algorithm parameters
- **Sequence configs** (`Config/Sequence/`) — dataset paths, camera intrinsics, stereo baseline
- **Train configs** (`Config/Train/`) — training parameters

### Key Data Structures

- **`StereoData`** (`DataLoader/Interface.py`) — one stereo pair: `imageL/R`, `K` matrix, `baseline`, `T_BS` extrinsics
- **`StereoFrame`** (`DataLoader/Interface.py`) — a timestamped frame containing one `StereoData` + optional ground truth
- **`VisualMap`** (`Module/Map/VisualMap.py`) — the global factor graph with three node types (`FrameStore`, `MatchStore`, `PointStore`) and directed edges connecting them (`frame2match`, `match2frame1/2`, `match2point`, `point2match`, `frame2map`)

## Critical Data Flow (per-frame)

1. **Frontend** receives `(frame_t1, frame_t2)` stereo pairs → outputs `depth1` + `depth_cov1` (from stereo disparity) and `match01` + `flow_cov` (from temporal matching), both via a single batched network forward pass
2. **KeypointSelector** picks N points on frame0 using a covariance-aware quality map (low uncertainty areas preferred)
3. **Covariance2to3** projects each 2D keypoint's depth/flow covariance into a full 3×3 spatial covariance matrix in camera frame, then rotates to world frame
4. **OutlierFilter** removes observations with degenerate covariance or geometric inconsistency
5. Observations are registered into `VisualMap` as `MatchObs` connecting `FrameNode` → `PointNode`
6. **Optimizer** (TwoFrame PGO) runs Levenberg-Marquardt with `weight = Σ⁻¹` (inverse covariance as information matrix) so uncertain observations are automatically down-weighted
7. Optimized pose written back to `VisualMap.frames`

The main loop lives in `Odometry/MACVO.py:run()` → `run_pair()`.

### Frontend Batching Strategy

`FlowFormerCovFrontend.estimate_pair()` batches stereo matching and temporal matching into one forward pass: `input_A = cat([imageL_t2, imageL_t1])`, `input_B = cat([imageR_t2, imageL_t2])`. Flow channels are then split: `flow[0]` = disparity → depth, `flow[1]` = optical flow → matching.

## Coordinate Conventions

- **Pixels**: `(u, v)` OpenCV convention (east-down). Index arrays as `data[..., v, u]`
- **World**: NED (`+x` North, `+y` East, `+z` Down), first frame at identity SE3
- **PyTorch**: `B×C×H×W` image tensors, batch dimension first
- **Lie groups**: `pypose` SE3 as 7-element tensors `[quat_xyzw, trans_xyz]`

## Optimization: Sequential vs Parallel

`IOptimizer` supports two modes controlled by `config.parallel`:
- **Sequential**: blocking optimization in the main thread
- **Parallel**: optimization runs in a spawned subprocess via `multiprocessing.Pipe`. `start_optimize()` sends data (non-blocking), `write_map()` receives result (blocks if not done). This lets the frontend process the next frame while the previous frame's PGO runs concurrently.

## Dependencies & Setup

- Python ≥ 3.10 (uses `match` syntax and new type annotations)
- CUDA ≥ 12.4, VRAM ≥ 6 GB (2.7 GB in fast mixed-precision mode)
- Git submodules: `Module/Network/FlowFormer`, `Baseline/DPVO` — clone with `--recursive`
- Pretrained models go in `Model/` directory (see README for download links)
- Docker build: `docker build --network=host -t macvo:latest -f Docker/Dockerfile .`
