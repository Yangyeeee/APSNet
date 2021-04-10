
## Classification

### Run
```
## train backbone

python train_cls.py --model pointnet_cls --log_dir pointnet --gpu 1 --sess pointnet128 -b 128 --lr 0.01 --epoch 400

```


```
## train samplenet

python train_samplenetarm.py --model pointnet_cls --log_dir pointnet --gpu 4 --l0 20000 --sampler samplenet --sess arm20000 --ar --lr 0.005 --epoch 400 -b 128
```


