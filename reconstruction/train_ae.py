"""
Author: Benny
Date: Nov 2019
"""

import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
# import provider
import importlib
# import shutil
import time
from time import localtime
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
from data.in_out import (
    snc_category_to_synth_id,
    create_dir,
    PointCloudDataSet,
    load_and_split_all_point_clouds_under_folder,
)

import torchvision
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('-b','--batch_size', type=int, default=50, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='point_ae', help='model name [default: point_ae]')
    parser.add_argument('--epoch',  default=500, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--sess', type=str, default="default", help='session')
    parser.add_argument('--l2', type=float, default=0, help='decay rate [default: 1e-4]')
    parser.add_argument('--object_class', type=str, default='multi',help='Single class name (for example: chair) or multi [default: multi]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--datafolder',  type=str, help='dataset folder')
    parser.add_argument("-in", "--num-in-points", type=int, default=2048, help="Number of input Points [default: 1024]")
    # For testing
    parser.add_argument('--test', action='store_true', help='Perform testing routine. Otherwise, the script will train.')
    return parser.parse_args()

def get_datasets(args):
    transforms = torchvision.transforms.Compose([PointcloudToTensor()])

    # define basic parameters
    project_dir = osp.dirname(osp.abspath(__file__))
    top_in_dir = osp.join(
        project_dir, "data", "shape_net_core_uniform_samples_2048"
    )  # top-dir of where point-clouds are stored., OnUnitCube()

    if args.object_class == "multi":
        class_name = ["chair", "table", "car", "airplane"]
    else:
        class_name = [str(args.object_class)]
    # load Point-Clouds
    syn_id = snc_category_to_synth_id()[class_name[0]]
    class_dir = osp.join(top_in_dir, syn_id)
    pc_data_train, pc_data_val, pc_data_test = load_and_split_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending=".ply", verbose=True
    )

    for i in range(1, len(class_name)):
        syn_id = snc_category_to_synth_id()[class_name[i]]
        class_dir = osp.join(top_in_dir, syn_id)
        (
            pc_data_train_curr,
            pc_data_val_curr,
            pc_data_test_curr,
        ) = load_and_split_all_point_clouds_under_folder(
            class_dir, n_threads=8, file_ending=".ply", verbose=True
        )
        pc_data_train.merge(pc_data_train_curr)
        pc_data_val.merge(pc_data_val_curr)
        pc_data_test.merge(pc_data_test_curr)

    if args.object_class == "multi":
        pc_data_train.shuffle_data(seed=55)
        pc_data_val.shuffle_data(seed=55)
    pc_data_train.transforms = transforms
    pc_data_val.transforms = transforms
    pc_data_test.transforms = transforms
    return pc_data_train, pc_data_val,pc_data_test

def test(model, loader):
    mean_loss = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, _ = data

        points = points.transpose(2, 1)
        points  = points.cuda()
        model = model.eval()
        rec = model(points)
        loss = model.get_loss(points.transpose(2,1),rec)
        mean_loss.append(loss.item())
    loss = np.mean(mean_loss)
    return loss

def eval(args):
    _, _ , testset= get_datasets(args)
    testDataLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    MODEL = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))str(experiment_dir) + '/checkpoints/best_model.pth'

    PointAE = MODEL.get_model().cuda()

    try:
        experiment_dir = Path('./log/')
        checkpoint =torch.load( "./log/pointae/checkpoints/best_model.pth")
        start_epoch = 0 #checkpoint['epoch']
        PointAE.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model,epoch {}'.format(start_epoch))
    except:
        print('No existing model, starting training from scratch...')
    with torch.no_grad():
        model = PointAE.eval()
        mean_loss = []
        for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points, _ = data
            points = points.transpose(2, 1)
            points  = points.cuda()
            rec = model(points)
            loss = model.get_per_loss(points.transpose(2,1),rec)
            mean_loss.append(loss)
        loss_per = torch.cat(mean_loss)
        np.save("loss_per.npy",loss_per.detach().cpu().numpy())
        loss = torch.mean(loss_per)
        print('Test loss', loss.item())
    return loss.item()
    # loss_per_pc_ref = np.load(file_path_ref)
    # log_file.write("Normalized reconstruction error: %.3f\n" % nre_per_pc.mean())



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    # experiment_dir = experiment_dir.joinpath('classification')
    # experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    #args = parse_args()
    current_time = time.strftime('%d_%H:%M:%S', localtime())
    writer = SummaryWriter(log_dir='runs/' + current_time+"_" + args.sess, flush_secs=30)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # args.datafolder = 'modelnet40_ply_hdf5_2048'

    trainset, testset,_  = get_datasets(args)
    # dataloader
    testDataLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    '''MODEL LOADING'''
    # num_class = 40
    MODEL = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    PointAE = MODEL.get_model().cuda()
    start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            PointAE.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.l2
        )
    else:
        optimizer = torch.optim.SGD(PointAE.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best =1000
    mean_loss = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, _ = data
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            # points = torch.Tensor(points)
            #

            points = points.transpose(2, 1)
            points = points.cuda()
            optimizer.zero_grad()

            PointAE = PointAE.train()
            recon = PointAE(points)
            loss = PointAE.get_loss(points.transpose(2,1),recon)
            mean_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        loss = np.mean(mean_loss)
        writer.add_scalar('loss/train_ae', loss, epoch)
        print('Train loss', loss.item())

        with torch.no_grad():
            loss = test(PointAE.eval(), testDataLoader)
            print('Test loss', loss.item())
            writer.add_scalar('loss/test_ae', loss, epoch)
            if (loss <= best):
                best = loss
                best_epoch = epoch + 1

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'loss': loss,
                    'model_state_dict': PointAE.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    writer.close()
    return loss

if __name__ == '__main__':
    args = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from src.pctransforms import OnUnitCube, PointcloudToTensor

    if args.test == 0:
        loss = []
        sess = args.sess
        args.sess =  "point_ae_lr{}".format(args.lr)
        res = main(args)
        loss.append(res)
        print(loss)
    else:
        loss = eval(args)
        print(loss)

