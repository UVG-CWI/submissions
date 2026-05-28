\# XJBG — Submission by Team Rewind



\## Team Name

Rewind



\## Team Members



| Name | Affiliation |

|------|-------------|

| Dayou Zhang | Capital Normal University |

| Yi Song | Capital Normal University |

| Ziyue Wan | Capital Normal University |

| Zeqiang Wei | Capital Normal University |



\## Algorithm Name

XJBG



\## Algorithm Description

This Python script is designed to extract timestamp-filtered synchronized RGB frames from 8 RealSense bag files and generate SAM3 foreground masks for each image, with a structured output layout that organizes frames and masks into numbered directories alongside critical metadata files (cameras.json and alignment\_metadata.json). It first reads start and end timestamps from a sequence JSON file to filter frames, verifies consistent frame counts across all cameras, and aligns frames by list index rather than frame number—ensuring synchronization for multi-camera setups. The script also leverages the pyrealsense2 library to extract camera serial numbers, color intrinsics (fx, fy, cx, cy, resolution), and extrinsics (c2w/w2c transforms) from bag files, which are essential for downstream 3D reconstruction tasks.



An optional Stage 3 of the workflow enables masked foreground point-cloud reconstruction using Depth Anything 3 (DA3), integrating RealSense depth data to refine DA3’s depth predictions via robust RANSAC-based scale/shift alignment and multi-view reprojection consistency filtering. This stage includes advanced features like SIFT feature detection and matching for correspondence-based depth refinement, epipolar error checking, triangulation of 3D points from matched features, and inverse distance weighting (IDW) for residual correction to improve depth accuracy. The reconstructed point clouds are downsampled (voxel filtering) and outlier-removed, then saved in formats compatible with Open3D, making the pipeline end-to-end for multi-camera RGB-D data processing and 3D reconstruction.



\## Processing Track

\*\*Full Pipeline\*\* (processing from raw `.bag` files)



\## How to Run



\### 1. Environment Setup



A conda environment file is provided. Create and activate the environment with:



```bash

conda env create -f environment.yml

conda activate <env\_name>

```



The following pre-trained models must be available before running:

\- `sam3\_modelscope` — SAM3 segmentation model

\- `DA3NESTED-GIANT-LARGE-1.1` — Depth Anything 3 model, expected at `/workspace/Depth-Anything-3/DA3NESTED-GIANT-LARGE-1.1`



\### 2. Run the Full Pipeline



```bash

python extract\_synced\_frames\_and\_maskes3.py \\

&#x20; --bag-dir /dataset/BlueSpeech/consumer-grade\_capture\_system/camera\_output \\

&#x20; --camera-config /dataset/BlueSpeech/consumer-grade\_capture\_system/camera\_output/BlueSpeech\_camera\_config.json \\

&#x20; --sequence-json /workspace/dataset\_sequence.json \\

&#x20; --seq-name BlueSpeech \\

&#x20; --output /workspace/BlueSpeech/frames \\

&#x20; --sam3-model sam3\_modelscope \\

&#x20; --device cuda \\

&#x20; --da3-model-path /workspace/Depth-Anything-3/DA3NESTED-GIANT-LARGE-1.1 \\

&#x20; --da3-device cuda \\

&#x20; --stage3-reconstruct

```



\## Hardware / Environment

\- \*\*GPU:\*\* NVIDIA RTX 4090 (24 GB VRAM)

\- \*\*OS / Runtime:\*\* See `environment.yml` for the full conda environment specification.



\## Runtime

Approximately \*\*\~3 min per frame\*\*.

