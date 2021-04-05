## Installation

### update gcc to 7.0
```bash
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*

then execute after start every terminal
scl enable devtoolset-7 bash

or add 
source /opt/rh/devtoolset-7/enable
to .bashrc file

```

### create env

```bash
conda create --name my_env python=3.7
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0=cuda10.0_0
```
#### install requirments
```bash
pip install -r requirement.txt
pip install -r req.txt
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Usage
### Data preparation
Create the 'car' dataset (ModelNet40 data will automatically be downloaded to `data/modelnet40_ply_hdf5_2048` if needed) and log directories:
```bash
python data/create_dataset_torch.py
```


### Training and evaluating



### Train SampleNet
To train orginal  SampleNet (with sample size 64 in this example), using an existing PCRNet as the task network, use:
```bash
python mainorg.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 --epochs 400
```

### Train l0 arm SampleNet

```bash
python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 --l0 10 --skip-projection --sess 10simp0k5b1 --lmbda 0 --alpha 0 --k1 5 --gpu 4 --lr 5e-4 --epochs 400
```
