# SERP-Net

We  present SeRP, a framework   for   Self-SupervisedLearning  of  3D  point  clouds.   SeRP  consists  of  encoder-decoder  architecture  that  takes  perturbed  or  corruptedpoint clouds as its inputs and aims to reconstruct the original  point  cloud  without  corruption. The  proposed  framework also  addresses  some  of  the  limitations  of  Transformers based [Masked Autoencoders](https://github.com/Pang-Yatian/Point-MAE) which are prone to leak-age of location information and uneven information density. Furthermore,  we  also  proposed  VASP:Vector-Quantized Autoencoder  for Self-supervised  Representation  Learning for Point Clouds that employs Vector-Quantization for discrete representation learning for Transformer based autoencoders. [arXiv](https://arxiv.org/pdf/2209.06067.pdf).

<img src="https://github.com/gargsid/SERPNet-Point-Cloud-Representation-Learning/blob/main/assets/vq-vae-pipeline.png" width="900" height="200" />
<img src="https://github.com/gargsid/SERPNet-Point-Cloud-Representation-Learning/blob/main/assets/reconstructions.png" width="750" height="300" />

## Setup
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  
cd extensions/chamfer_dist  
python setup.py install --user  
cd ../../  
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install tqdm wandb pandas opencv-python matplotlib timm
```

## Links to download data and trained models

Data link: https://drive.google.com/file/d/18fWjXj2Io6kcHOe9DKnInFliAvveKJMA/view?usp=sharing

SeRP-PointNet Saved Models: https://drive.google.com/file/d/1bn8XhYt4e6UCklfsQv-Tikk_kDPOwswV/view?usp=sharing

SeRP-Transformer Saved Models: https://drive.google.com/file/d/12LJmrf5AyBxlqZWx6Bn4UtXWUIJ66rPL/view?usp=sharing

Link to presentation video: https://drive.google.com/file/d/1G_3EZLqnKXF9Gj7t_dZB_64rqGyBicUM/view?usp=sharing

Link to the whole code + presentation video + presentation + report .zip: https://drive.google.com/drive/folders/1ZY8IguYaUMsaQsc-VFTUmPVl2B49OBCv?usp=sharing

<!-- ## Project Structure:  
Extract the data and saved models folders after downloading from the link above. The final folder structure should be the following:
```
|  
|-- data/
|         |--ModelNet/
|         |     |--modelnet40_train_8192pts_fps.dat
|         |     |--modelnet40_test_8192pts_fps.dat
|         |  
|         |--ShapeNet55/
|         |     |--ShapeNet55/
|         |     |       |--label_ids.pth
|         |     |       |--train_split.csv
|         |     |       |--val_split.csv
|         |     |       |--shapenet_pc/
|         |     |       |       |--pc_1.npy
|         |     |       |       |--pc_2.npy
|         |     |       |       |--.....npy
|         |     |       |       |--.....npy
|         |     |       |       |--pc_N.npy
|         |         
|
|-- SeRP-PointNet/
|         |
|         |--saved_models_pointnet/
|         | ... <remaining files>
|
|-- SeRP-transformer
|         |--models/
|         |     |--pre-trained/  # pre-trained models on ShapeNet-55
|         |     |   |--tr_serp/     # pre-trained SeRP-Transformer
|         |     |   |--tr_vasp/     # pre-trained VASP-Transformer
|         |     |   
|         |     |--fine-tuned/  # fine-tuned models on classification tasks
|         |     |   |--m40_no_pretrain/  # encoder trained from scratch on ModelNet40 dataset
|         |     |   |--m40_tr_serp/  # Pre-trained SeRP encoder on ModelNet40 dataset
|         |     |   |--m40_tr_serp/  # Pre-trained VASP encoder on ModelNet40 dataset
|         |     |   |--sh_tr_nopretrain/  # encoder trained from scratch on ShapeNet classification dataset
|         |     |   |--sh_tr_serp/  # Pre-trained SeRP encoder on ShapeNet55 dataset
|         |     |   |--sh_tr_vasp/  # Pre-trained VASP encoder on ShapeNet55 dataset
|         |     |   
|         |     |--pretrain/  # by default to store pretrained models while pre-training
|         |     |   --model.pth  # this will be written here
|         |     |   --logs.txt  # this will be written here
|         |     |   
|         |     |--finetune/  # by default to store classification models while finetuning
|         |     |   --model.pth  # this will be written here
|         |     |   --logs.txt  # this will be written here
|
|-- extensions/
``` -->

# How to run 

## SeRP Transformer

### Requirements

```PyTorch >= 1.7.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;```

### Pre-training on ShapeNet

#### SeRP-Transformer

```
python pretrain.py --logs_dir=<LOGS_DIR to store pretrained model and logs.txt file> --loss_type=cdl2 --batch_size=128
# model.pth and logs.txt file will be stored as checkpoints in the LOGS_DIR provided
```

#### VASP-Transformer

```
python pretrain.py --logs_dir=<LOGS_DIR to store pretrained model and logs.txt file> --batch_size=64  --use_vq=True
# model.pth and logs.txt file will be stored as checkpoints in the LOGS_DIR provided
```

### Fine-Tune on downstream task 

Training the transformer encoder without pretrained model

```
# In the this command, logs_dir is the path to LOGS_DIR where you want to store the trained model. 
# Use dataset=shapenet to train on ShapeNet55 dataset or dataset=modelnet to train on ModelNet40 dataset

python finetune.py \
--logs_dir=LOGS_DIR \
--epochs=300 \
--dataset=modelnet \
--learning_rate=0.0001 \
--weight_decay=0.001

python finetune.py \
--logs_dir=LOGS_DIR \
--epochs=300 \
--dataset=shapenet \
--learning_rate=0.0001 \
--weight_decay=0.001
```

Training the transformer encoder with pretrained model

```
# SeRP-Transformer
python finetune.py \
--logs_dir=LOGS_DIR \ # directory in which the trained model is saved
--prev_ckpt=models/pre-trained/tr_serp/model.pth \ # path to the pre-trained transformer ckpt
--epochs=300 \
--dataset=modelnet \  # can use dataset=modelnet or dataset=shapenet
--learning_rate=0.0001 \
--weight_decay=0.001 

# VASP-Transformer
python finetune.py \
--logs_dir=LOGS_DIR \ # directory in which the trained model is saved
--prev_ckpt=models/pre-trained/tr_vasp/model.pth \  # path to the pre-trained transformer ckpt
--epochs=300 \
--dataset=modelnet \  # can use dataset=modelnet or dataset=shapenet
--learning_rate=0.0001 \
--weight_decay=0.001 \

DATASET = {modelnet, shapenet}
```

#### Evaluating fine-tuned models

```
# Transformer trained on ModelNet40 with random weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_no_pretrain/model.pth

# Transformer trained on ShapeNet55 with random weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_nopretrain/model.pth

# SeRP-Transformer trained on ModelNet40 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_tr_serp/model.pth

# SeRP-Transformer trained on ShapeNet55 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_serp/model.pth

# VASP trained on ModelNet40 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/m40_tr_vasp/model.pth

# VASP trained on ShapeNet55 with pre-trained weights
python evaluate_classification.py \
--finetuned_model=models/fine-tuned/sh_tr_vasp/model.pth
```

Note: Instructions for running the SeRP-PointNet is are provided in ```SeRP-PointNet/README.md``` directory

## Source code citations and Acknowledgments
Some parts of `SeRP-PointNet/pointnet.py` were sourced from https://github.com/fxia22/pointnet.pytorch

Transformer model was adapted from [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

Vector-Quantization operations and gradient operations were adapted from [VQ-VAE](https://github.com/jaywalnut310/Vector-Quantized-Autoencoders)

Processed datasets for ShapeNet55 and ModelNet40 are taken from [Point-BERT](https://github.com/lulutang0608/Point-BERT/tree/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52)

## Contributions

[Siddhant Garg](https://gargsid.github.io/):
```
Implemented by Siddhant Garg 
  |--SeRP-transformer/data_utils.py
  |--SeRP-transformer/evaluate_classification.py
  |--SeRP-transformer/finetune.py
  |--SeRP-transformer/generate_vasp.py
  |--SeRP-transformer/plot_tsne.py
  |--SeRP-transformer/reconstruct.py
  |--SeRP-transformer/utils.py
  |--SeRP-transformer/serp_transformer.py 
  |--SeRP-transformer/vq_vae.py 
```

[Mudit Chaudhary](https://github.com/muditchaudhary/SERP-Net)
```
SeRP-PoinNet/models/pointnet.py
SeRP-PointNet/eval.py  
SeRP-PointNet/finetune.py <with Siddhant Garg>
SeRP-PointNet/pretrain.py <with Siddhant Garg>
```
