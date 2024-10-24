## Dependencies and Installation

1. Navigate to segmentation folder

   ```bash
   cd DreamClear/segmentation
   ```

2. Create Conda Environment and Install Package
   ```bash
   conda create -n rmt_seg python=3.9 -y
   conda activate rmt_seg
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   pip3 uninstall mmseg
   python3 setup.py install
   pip3 install importlib_metadata 
   ```
3. Put pre-trained [rmt_uper_s_2x.pth](https://huggingface.co/shallowdream204/DreamClear/blob/main/rmt_uper_s_2x.pth) into `./ckpt/`.

## Dataset Preparation

1. Download ADE20K validation set from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/). 

2. Put the image restoration results into `./data/val_data/`. The directory structure should look like

   ```
    data/
    │
    ├── val_ann/
    │   ├── ADE_val_00000001.png
    │   ├── ADE_val_00000002.png
    │   └── ...
    │
    ├── val_data/
        ├── ADE_val_00000001.jpg
        ├── ADE_val_00000002.jpg
        └── ...
   ```

## Evaluation
Run the following commands to get the segmentation results
```
bash tools/dist_test.sh configs/RMT/RMT_Uper_s_2x.py ckpt/rmt_uper_s_2x.pth 8 \
--options data.test.data_root='data' data.test.img_dir='val_data' \ 
data.test.ann_dir='val_ann' data.samples_per_gpu=4  --eval mIoU
# test_image full_path is data.test.data_root/data.test.img_dir; ann_dir is data.test.data_root/data.test.ann_dir
# so make a link to your data path under the 'data' path
# 8 is the number of GPUs, can change larger
# data.samples_per_gpu is the batchsize
```
