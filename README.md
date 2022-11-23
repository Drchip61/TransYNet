# Fully-Transformer-Network-for-Change-Detection-of-Remote-Sensing-Images

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./dataset/README.md"](datasets/README.md) for details, or please send an Email to 2981431354@mail.dlut.edu.cn to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@
  title={Swin Ynet},
  author={Tianyu Yan, Fuzi Wan},
  journal={投哪一篇呢嘿嘿},
  year={2022}
}
```
