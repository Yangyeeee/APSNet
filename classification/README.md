
## Classification

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

python train_samplenetarm.py --model pointnet_cls --log_dir pointnet --gpu 4 --l0 20000 --beta 0.01 --sampler samplenet --sess arm20000 --ar --lr 0.001 --epoch 400 -b 128
```


