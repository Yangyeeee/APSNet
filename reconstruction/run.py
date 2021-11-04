import os
import pickle
import torch
import argparse
parser = argparse.ArgumentParser(description='GAT')
# parser.add_argument('--dataset', type=str, default="wisconsin", help='dataset.')
# parser.add_argument('--h', type=int, default=96, help='dataset.')
# parser.add_argument('--lr', type=float, default=0.05, help='dataset.')
# parser.add_argument('--l2', type=float, default=5e-4, help='dataset.')
# parser.add_argument('--l0', type=float, default=1e-1, help='dataset.')
# parser.add_argument('--idrop', type=float, default=0, help='dataset.')
parser.add_argument('--gpu', type=int, default=2, help='gpu.')
args = parser.parse_args()
# file = "result_{}.pkl".format(args.dataset)

# os.system("rm -f {}".format(file))
# for i in range(5):
#     j = [ 2,4,6,8,10]
#     c = "python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 " \
#         "--l0 {} --skip-projection --sess arml05e-2t{}simp100k1 --lmbda 0 --gpu 4 --alpha 100 --epochs 400 --k1 1 --a {}".format(5e-2, j[i],j[i])
#     print(c)
#     os.system(c )   #96, teaxs 48 idrop 0 0.01,0.05,0.1,,4,4.5,5,5.5,6,6.5,7

# for i in range(3):
#         a = [32,16,8]
#         c = "python train_samplenetorg.py --model pointnet_cls --log_dir pointnet --gpu 2 --sampler samplenet --sess sampleorg{} --lr 0.01 --epoch 400 -b 128 --num_out_points {}".format(a[i],a[i], args.gpu)
#         print(c)
#         os.system(c)

# for i in range(10):
#         a = [1,2,3,4 ,5,6,7,8,9,10]
#         c = "python train_samplenetarm.py --model pointnet_cls --log_dir pointnet  --l0 1 --sampler samplenet --k1 {} --sess armk{} --ar --lr 0.01 --epoch 400 --gpu {}".format(a[i],a[i], args.gpu)
#         print(c)
#         os.system(c)

# for i in range(10):
#         a = [1,2,3,4 ,5,6,7,8,9,10]
#         c = "python train_samplenetarm.py --model pointnet_cls --log_dir pointnet  --l0 {} --beta 0.01 --sampler samplenet --sess ar{}nofill --k 5  --lr 0.001 --epoch 400 -b 128 --ar --gpu {}".format(a[i]*10000,a[i]*10000, args.gpu)
#         print(c)
#         os.system(c)

for i in range(5):
        a = [ 1,5,10,50,100]
        c = "python testgreedy.py --beta {} --sess {} --batch_size 512 --gpu {}".format(a[i],a[i], args.gpu)
        print(c)
        os.system(c)



# c = "python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 " \
#         "--l0 5e-2 --skip-projection --sess arml05e-2t10simp10k1 --lmbda 0 --gpu 1 --alpha 10 --epochs 400 --k1 1 --a 10"
# print(c)
# os.system(c)
#
# c = "python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 " \
#         "--l0 5e-2 --skip-projection --sess arml05e-2t5simp10k1 --lmbda 0 --gpu 1 --alpha 10 --epochs 400 --k1 1 --a 5"
# print(c)
# os.system(c)
#
# c = "python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 " \
#         "--l0 5e-2 --skip-projection --sess arml05e-2t5simp5k1 --lmbda 0 --gpu 1 --alpha 5 --epochs 400 --k1 1 --a 5"
# print(c)
# os.system(c)
# for i in range(8):
#     j = [ 0.001, 0.1,1,1.5,2,2.5,3,3.5]
#     c = "python mainarm.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64 " \
#         "--l0 5e-3 --skip-projection --sess ar5e-3t50simp100k{} --lmbda 0 --gpu 3 --alpha 100 --epochs 400 --k1 {}".format(j[i], j[i])
#     print(c)
#     os.system(c )   #96, teaxs 48 idrop 0


# if  os.path.exists(file):
#     f = open(file, 'rb')
#     a = pickle.load(f,encoding="bytes")
#     print("average result",torch.tensor(a).mean())
#     f.close()
#     os.system("rm -f {}".format(file))