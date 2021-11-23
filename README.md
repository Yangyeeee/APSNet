# APSNet: Attention Based Point Cloud Sampling

## Installation

### 1. Update gcc to 7.0

### 2. Create env

```bash
conda create --name my_env python=3.7
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0=cuda10.0_0
```

### 3. Install requirments
```bash
pip install -r requirement.txt
pip install -r req.txt
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Usage

### 1. Classification
```
## train apsnet 
python train_aps.py  --gpu 4 --epoch 400 -b 128 --num_out_points 32

## train task network
python train_cls.py --model pointnet_cls --log_dir pointnet --gpu 1  -b 128 --lr 0.01 --epoch 400

```

### 2. Reconstruction
```
## train apsnet 
python train_aps.py  --gpu 4 --epoch 400 -b 128 --num_out_points 32

## train task network
python train_ae.py --model point_ae --log_dir pointae --gpu 1  -b 128 --lr 0.0001 --epoch 400

```


### 3. Registration
```
## Data preparation
Create the 'car' dataset (ModelNet40 data will automatically be downloaded to `data/modelnet40_ply_hdf5_2048` if needed) and log directories:
python data/create_dataset_torch.py


## train apsnet 
python train_aps.py  --gpu 4 --epoch 400 -b 128 --num_out_points 32


## Train *PCRNet* (supervised) registration network
To train a *PCRNet* model to register point clouds, use:
python main.py -o log/baseline/PCRNet1024 --datafolder car_hdf5_2048 --sampler none --train-pcrnet --epochs 500


## Train SampleNet
To train SampleNet (with sample size 64 in this example), using an existing PCRNet as the task network, use:
python main.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64


```
