# AWABS
** Long-Tailed Recognition**


**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

**Stage-1**:

To train a model for Stage-1 with *mixup(CutOut, CutMix, Cut-Thumbnail)*, run:

(one GPU for CIFAR-10-LT & CIFAR-100-LT)

```
python3 train_stage1.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml
```

`DATASETNAME` can be selected from `cifar10`,  `cifar100`.

`ARCH` can be `resnet32` for `cifar10/100`.

**Stage-2**:

To train a model for Stage-2 with *one GPU* (all the above datasets), run:

```
python3 train_stage2.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_awabs.yaml resume /path/to/checkpoint/stage1
```

The saved folder (including logs and checkpoints) is organized as follows.
```
AWABS
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```
**Example :**
```
Stage 1: CIFAR10_LT
python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb01_stage1_mixup.yaml
After stage 1: cifar10_imb01_stage1_mixup_202402010007    

Stage 2:
python3 train_stage2.py --cfg ./config/cifar10/cifar10_imb01_stage2_mislas.yaml resume ./saved/cifar10_imb01_stage1_mixup_202402010007/ckps/model_best.pth.tar
After stage 2: cifar10_imb01_stage2_awabs_202402010032
```
## Evaluation

To evaluate a trained model, run:

```
python3 eval.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage1_mixup.yaml  resume /path/to/checkpoint/stage1
python3 eval.py --cfg ./config/DATASETNAME/DATASETNAME_ARCH_stage2_mislas.yaml resume /path/to/checkpoint/stage2
```

## Contact

If you have any questions about our work, feel free to contact us through email (sasankaj.math@gmail.com) 
