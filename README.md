###  BIM-VG


## Requirements

```shell
conda create -n BiMS-VG python=3.10
conda activate BiMS-VG
pip install requirements.txt
```

## Getting Started
Please refer to [GETTING_STARGTED.md](GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints.



## **File Structure**

```
data/
├── gref/
│   ├── corpus.pth
│   ├── grey_train.pth
│   └── grey_val.pth
├── images/
│   ├── train2014/
│   └── val2014/
├── unc/
│   ├── corpus.pth
│   ├── unc_testA.pth
│   ├── unc_testB.pth
│   ├── unc_train.pth
│   ├── unc_trainval.pth
│   └── unc_val.pth
```

```
bert-base-uncased/
├── config.json
├── model.safetensors
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt
```

```
project/
├── bert-base-uncased/           # 预训练 BERT 模型文件
├── data/                        # 数据集
├── datasets/                    # 数据处理模块
├── models/                      # 自定义模型定义
├── outputs/                     # 训练输出和日志
├── checkpoints/                 # 模型检查点保存
├── utils/                       # 工具函数和脚本
├── engine.py                    # 训练/验证引擎核心 
├── eval.py                      # 模型评估脚本 
└── train.py                     # 主训练脚本 
```



## Training

```shell
# refcocog-g
python  train.py --aug_crop --aug_scale --aug_translate --detr_model ./checkpoints/detr-r50-gref.pth --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 
# refcocog-u
python train.py --aug_crop --aug_scale --aug_translate  --detr_model ./checkpoints/detr-r50-gref.pth - --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50 
# refcoco
python train.py  --aug_crop --aug_scale --aug_translate  --detr_model ./checkpoints/detr-r50-unc.pth ---dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 
# refcoco plus
python train.py  --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50
```

## Inference
[Checkpoints](https://drive.google.com/drive/folders/1stGPq4Sz_Vu60QliUzey8m6iYXrrF3Ua?usp=drive_link)

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# refcocog-g
python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --model_type ResNet --batch_size 16 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --eval_model outputs/refcocog_gsplit_r50/best_checkpoint.pth --eval_set val
```



## Acknowledgments
This project is built upon [Dyanmic MDETR](https://github.com/djiajunustc/TransVG). Thanks for their wonderful work!

## Contact

a1610874650@gmail.com