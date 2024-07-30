# Getting Started
We follow [StreamPETR](https://github.com/exiawsh/StreamPETR/tree/main/docs) to set up environmnet, prepare data, train, inference and visualize.

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Train & Inference](#train--inference)
- [Visualize](#visualize)
## Environment Setup

### Base Environments
Python >= 3.8 \
CUDA == 11.1 \
PyTorch == 1.9.0 \
mmdet3d == 1.0.0rc6 \
[flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2


### Step-by-step Installation Instructions
**a. Create a conda virtual environment and activate it.**
```shell
conda create -n dvpe python=3.8 -y
conda activate dvpe
```

**b. Install PyTorch and torchvision.**
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
**c. Install flash-attn (optional).**
```
pip install setuptools==59.5.0
pip install flash-attn==0.2.2
```

**d. Clone DVPE.**
```
git clone https://github.com/dop0/DVPE
```

**e. Install mmdet3d.**
```shell
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd ./DVPE
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
pip install IPython
# downgrade version
pip install numpy==1.23.5
pip install yapf==0.40.0
```
## Data Preparation

### Dataset
Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

### Creating Infos File

Compared to data preparation in `MMDetection3D`, 2D annotations and temporal information are additionally created for training/evaluation.
```shell
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes2d --version v1.0
```

Using the above code will generate `nuscenes2d_temporal_infos_{train,val,test}.pkl`.
We also privided the processed [train](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_train.pkl), [val](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_val.pkl) and [test](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_test.pkl) pkl.

### Pretrained Weights
Please download the pretrained weights to `./ckpts`. In train set, we use the [R50 pre-trained on ImageNet](https://download.pytorch.org/models/resnet50-0676ba61.pth), [R50 pre-trained on nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) and [R101 pre-trained on nuImage](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth). In test set, we use the [DD3D pretrained VoVNet-99](https://github.com/exiawsh/storage/releases/download/v1.0/dd3d_det_final.pth).

### Folder Structure
After preparation, you will be able to see the following directory structure:
```
DVPE
├── projects/
├── mmdetection3d/
├── ckpts/
│   ├── xxx.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes2d_temporal_infos_train.pkl
|   |   ├── nuscenes2d_temporal_infos_val.pkl
├── ...
```
## Train & Inference
### Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh projects/configs/DVPE/dvpe_ablation_nui_704_bs4_24e_900q_gpu4.py 4 --work-dir work_dirs/dvpe_ablation_nui_704_bs4_24e_900q_gpu4/
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh projects/configs/DVPE/dvpe_ablation_nui_704_bs4_24e_900q_gpu4.py work_dirs/dvpe_ablation_nui_704_bs4_24e_900q_gpu4/latest.pth 2 --eval bbox
```
### Estimate the inference speed
The latency includes data-processing, network (FP32) and post-processing. Noting that \"workers_per_gpu\" may affect the speed because we include data processing time.
```bash
python tools/benchmark.py projects/configs/test_speed/dvpe_train_nui_704_bs4_60e_428q_gpu4_speed_test.py
```

## Visualize
You can generate the reault json following:
```bash
CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh projects/configs/DVPE/dvpe_ablation_nui_704_bs4_24e_900q_gpu4.py work_dirs/dvpe_ablation_nui_704_bs4_24e_900q_gpu4/latest.pth 2 --format-only
```
You can visualize the 3D object detection following:
```bash
python3 tools/visualize.py
# please change the results_nusc.json path in the python file
```