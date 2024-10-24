## Dependencies and Installation

1. Navigate to detection folder

   ```bash
   cd DreamClear/detection
   ```

2. Create Conda Environment and Install Package
   ```bash
   conda create -n rmt_det python=3.9 -y
   conda activate rmt_det
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   pip3 uninstall mmdet
   pip3 install openmim
   mim install mmengine
   pip3 uninstall mmcv
   pip3 install mmcv-full
   pip3 install numpy==1.23.5
   pip3 install cython==0.29.33
   pip3 uninstall pycocotools
   pip3 install mmpycocotools
   python3 setup.py install
   pip3 install terminaltables
   pip3 install efficientnet_pytorch
   ```
3. Put pre-trained [rmt_maskrcnn_s_1x.pth](https://huggingface.co/shallowdream204/DreamClear/blob/main/rmt_maskrcnn_s_1x.pth) into `./ckpt/`.

## Dataset Preparation

1. Download COCO 2017 validation set from the [official website](https://cocodataset.org/#download).

2. Put the image restoration results into `./data/val2017/`. The directory structure should look like

   ```
   data/
   │
   ├── instances_val2017.json
   │
   └── val2017/
      ├── 000000000139.jpg
      ├── 000000000285.jpg
      └── ...
   ```

## Evaluation
Run the following commands to get the segmentation results
```
bash tools/dist_test.sh configs/RMT/maskrcnn_s_1x.py \
ckpt/rmt_maskrcnn_s_1x.pth 8 --cfg-options \ 
data.test.ann_file='data/instances_val2017.json' \ 
data.test.img_prefix='data/val2017' --eval bbox segm
```
