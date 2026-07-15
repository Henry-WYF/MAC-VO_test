# MAC-VO DBoW2 provenance

- Source: `UZ-SLAMLab/ORB_SLAM3`, tag `v1.0-release`
- Commit: `0df83dde1c85c7ab91a0d47de7a29685d046f637`
- Imported paths: `Thirdparty/DBoW2/DBoW2`, `Thirdparty/DBoW2/DUtils`, and the accompanying README
- Local additions: this CMake file and `binding.cpp`; the upstream `-march=native` flag is intentionally not used
- License text: `LICENSE.txt`, obtained from the upstream DBoW2 `v1.1-free` release at commit `96d4276fec7da4ecb7ad9813b502367f5302884e`

The imported ORB-SLAM3 README states that all files in its DBoW2/DUtils
snapshot are BSD-licensed.  The DBoW2 license requires notifying the original
author when source or binary redistributions are made.  Complete that notice
before publishing or distributing a repository/image containing this code.

No ORB-SLAM3 `KeyFrameDatabase`, `LoopClosing`, map, g2o, or other GPL source is
included here.
