
## Classification


## Installation

### create env

```bash
conda create --name my_env python=3.7
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

#### install requirments
```bash
pip install kornia==0.3.0 tqdm==4.60.0 h5py==3.2.1 tensorboard==1.15.0
```


### Run
```
## train backbone

python train_cls.py --model pointnet_cls --log_dir pointnet_rerun --gpu 1 --sess pointnet128 -b 128 --lr 0.01 --epoch 400

```


```
## train samplenet original

python train_samplenetorg.py --model pointnet_cls --log_dir pointnet --gpu 4 --sampler samplenet --sess sampleorg --lr 0.01 --epoch 400 -b 128 --num_out_points 32
```


```
## train samplenet L0

python train_samplenetarm.py --model pointnet_cls --log_dir pointnet --gpu 4 --l0 20000 --beta 0.01 --k 1 --bias 0 --sampler samplenet --sess arm20000 --ar --lr 0.0001 --epoch 400 -b 128
```


