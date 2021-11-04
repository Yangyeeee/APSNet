
## Classification


## Installation

### create env

```bash
conda create --name my_env python=3.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

#### install requirments
```bash
pip install kornia tqdm h5py tensorboard
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

python train_samplenetarm.py --model pointnet_cls --log_dir pointnet --gpu 4 --l0 10000 --beta 0.01 --k 5 --bias 0 --sampler samplenet --sess ar10000 --ar --lr 0.001 --epoch 400 -b 128
```
```
## test greedy

python testgreedy.py --beta 500 --max
python testgreedy.py --fps --batch_size 256
```

