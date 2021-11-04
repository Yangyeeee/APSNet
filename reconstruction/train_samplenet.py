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
import provider
import importlib
import time
from time import localtime
from torch.utils.tensorboard import SummaryWriter

from data.modelnet_loader_torch import ModelNetCls
import numpy as np
import matplotlib.pyplot as plt
import random
import os.path as osp
from data.in_out import (
    snc_category_to_synth_id,
    create_dir,
    PointCloudDataSet,
    load_and_split_all_point_clouds_under_folder,
)
from src.general_utils import plot_3d_point_cloud
import torchvision
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('SSN')
    parser.add_argument('-b','--batch_size', type=int, default=50, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='point_ae', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=400, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training [sgd or adam default: adam]')
    parser.add_argument('--log_dir', type=str, default="pointnet", help='experiment root')
    parser.add_argument('--sess', type=str, default="default", help='session')
    parser.add_argument('--weight_decay', type=float, default=0, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='decay rate [default: 0.7]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')

    parser.add_argument('--sampler', default="samplenet", choices=['fps', 'samplenet', 'random', 'none'], type=str, help='Sampling method.')
    parser.add_argument('--train-samplenet', action='store_true', default=True,help='Allow SampleNet training.')
    parser.add_argument('--train_cls', action='store_true', default=False, help='Allow calssifier training.')
    parser.add_argument('--num_out_points', type=int, default=32, help='sampled Point Number [default: 32]')
    parser.add_argument('--projection_group_size', type=int, default=16, help='projection_group_size')
    parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck_size')
    parser.add_argument('--alpha', default=0.01, type=float, help='alpha')
    parser.add_argument('--lmbda', default=0.0001, type=float, help='lmbda')
    parser.add_argument('--datafolder',  type=str, help='dataset folder')
    parser.add_argument("-in", "--num-in-points", type=int, default=2048, help="Number of input Points [default: 1024]")
    parser.add_argument('--object_class', type=str, default='multi',
                        help='Single class name (for example: chair) or multi [default: multi]')
    # For testing
    parser.add_argument('--test', action='store_true', help='Perform testing routine. Otherwise, the script will train.')
    parser.add_argument('--vis', action='store_true', help='Visualize results [default: False]')
    return parser.parse_args()



def get_datasets(args):
    transforms = torchvision.transforms.Compose([PointcloudToTensor()])

    # define basic parameters
    project_dir = osp.dirname(osp.abspath(__file__))
    top_in_dir = osp.join(
        project_dir, "data", "shape_net_core_uniform_samples_2048"
    )  # top-dir of where point-clouds are stored., OnUnitCube()
    top_out_dir = osp.join(project_dir)  # use to save Neural-Net check-points etc.

    if args.object_class == "multi":
        class_name = ["chair", "table", "car", "airplane"]
    else:
        class_name = [str(args.object_class)]

    # experiment_name = "autoencoder"
    # n_pc_points = 2048  # Number of points per model
    # bneck_size = 128  # Bottleneck-AE size
    # ae_loss = "chamfer"  # Loss to optimize

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


def test(model, sampler, loader):
    mean_loss = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, _ = data
        points  = points.cuda()
        sampler = sampler.eval()
        p0_simplified, p0_projected = sampler(points)
        sampled_points = p0_projected.transpose(2, 1)

        model = model.eval()
        rec = model(sampled_points)
        loss = model.get_loss(points,rec)
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
    sampler = SampleNet(
        num_out_points=args.num_out_points,
        bottleneck_size=args.bottleneck_size,
        group_size=args.projection_group_size,
        initial_temperature=1.0,
        input_shape="bnc",
        output_shape="bnc",
        skip_projection=False,
    ).cuda()

    try:
        experiment_dir = Path('./log/')
        checkpoint =torch.load( "./log/0.0001/checkpoints/best_model.pth")
        start_epoch = checkpoint['epoch']
        PointAE.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model,epoch {}'.format(start_epoch))

        snet_checkpoint =torch.load( "./log/{}/checkpoints/best_samplenet.pth".format(args.log_dir))
        sampler.load_state_dict(snet_checkpoint['model_state_dict'])
        print('Use lstm sampler from {}'.format(args.log_dir))

    except:
        print('No existing model, starting training from scratch...')
    with torch.no_grad():
        model = PointAE.eval()
        loss = 0
        if args.vis:
            i = 2500
            points = torch.tensor(testDataLoader.dataset.point_clouds[i]).unsqueeze(0)
            points = points.cuda()
            sampler = sampler.eval()
            p0_simplified, p0_projected = sampler(points)
            sampled_points = p0_projected.transpose(2, 1)

            rec = model(sampled_points)
            loss = model.get_per_loss(points, rec)
            loss_per_ae = np.load("loss_per.npy")
            loss_n = loss.detach().cpu().numpy()/loss_per_ae[i]

            plot_3d_point_cloud(
                org = points[0].detach().cpu(),
                in_u_sphere=True, elev=77,azim=-90,
                title="Original point cloud",
            )
            plot_3d_point_cloud(
                p0_projected[0].detach().cpu(),points[0].detach().cpu(), c="#cfcfcf",in_u_sphere=True,  elev=77,azim=-90,title="Snet sampled points {}".format(args.num_out_points)
            )
            plot_3d_point_cloud(
                org = rec[0].detach().cpu(),
                in_u_sphere=True, elev=77,azim=-90,
                title="Snet Reconstruction {}, NRL {:.2f}".format(args.num_out_points,loss_n.item()),
            )

        else:
            mean_loss = []
            for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                points, _ = data
                points = points.cuda()
                sampler = sampler.eval()
                p0_simplified, p0_projected = sampler(points)
                sampled_points = p0_projected.transpose(2, 1)

                rec = model(sampled_points)
                loss = model.get_per_loss(points,rec)
                mean_loss.append(loss)
            loss_per = torch.cat(mean_loss)
            loss = torch.mean(loss_per)
            print('Test loss', loss.item())
            loss_per_ae = np.load("loss_per.npy")
            norma_loss = np.mean(loss_per.detach().cpu().numpy()/loss_per_ae)
            print('Normalzied Test loss', norma_loss)


    return loss
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
    # args = parse_args()
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

    trainset, valset,_ = get_datasets(args)
    # dataloader
    valDataLoader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    # TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
    #                                                  normal_channel=args.normal)
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
    #                                                 normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40
    MODEL = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    PointAE = MODEL.get_model().cuda()
    # criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load("./log/0.0001/checkpoints/best_model.pth")
        restore_epoch = checkpoint['epoch']
        PointAE.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model, epoch ' , restore_epoch)
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.train_cls:
        PointAE.requires_grad_(True)
        PointAE.train()
    else:
        PointAE.requires_grad_(False)
        PointAE.eval()

        # Create sampling network
    if args.sampler == "samplenet":
        sampler = SampleNet(
            num_out_points=args.num_out_points,
            bottleneck_size=args.bottleneck_size,
            group_size=args.projection_group_size,
            initial_temperature=1.0,
            input_shape="bnc",
            output_shape="bnc",
            skip_projection=False,
        ).cuda()

        if args.train_samplenet:
            sampler.requires_grad_(True)
            sampler.train()
        else:
            sampler.requires_grad_(False)
            sampler.eval()

    elif args.sampler == "fps":
        sampler = FPSSampler(
            args.num_out_points, permute=True, input_shape="bnc", output_shape="bnc"
        )

    elif args.sampler == "random":
        sampler = RandomSampler(
            args.num_out_points, input_shape="bnc", output_shape="bnc"
        )

    else:
        sampler = None

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            sampler.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.decay_rate)
    global_epoch = 0
    global_step = 0
    best = 100000.0
    loss_task = []
    loss_simple = []

    def compute_samplenet_loss(sampler, data, epoch):
        """Sample point clouds using SampleNet and compute sampling associated losses."""

        p0 = data
        p0_simplified, p0_projected = sampler(p0)

        # Sampling loss
        p0_simplification_loss = sampler.get_simplification_loss(p0, p0_simplified, args.num_out_points, 1, 0)

        simplification_loss = p0_simplification_loss
        sampled_data = (p0, p0_projected)
        # Projection loss
        projection_loss = sampler.get_projection_loss()

        samplenet_loss = args.alpha * simplification_loss + args.lmbda * projection_loss

        samplenet_loss_info = {
            "simplification_loss": simplification_loss,
            "projection_loss": projection_loss,
        }

        return samplenet_loss, sampled_data, samplenet_loss_info

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # scheduler.step()
        # if epoch%10 ==0 and epoch >0:
        #     sampler.temp *= 0.8
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            # points = torch.Tensor(points)
            # target = target[:, 0]

            points  = points.cuda()
            optimizer.zero_grad()
            sampler = sampler.train()
            sampler_loss, sampled_data, sampler_loss_info = compute_samplenet_loss(sampler, points, epoch)

            # classifier = classifier.train()
            # points = torch.cat((sampled_data[1],points[:,:,3:]),dim=-1)
            sampled_points = sampled_data[1].transpose(2, 1)
            recon = PointAE(sampled_points)
            loss_t = PointAE.get_loss(points, recon)
            loss_s = sampler_loss

            loss_task.append(loss_t.item())
            loss_simple.append(loss_s.item())
            (loss_t + loss_s).backward()
            optimizer.step()
            global_step += 1

        writer.add_scalar('loss/loss_task', np.mean(loss_task), epoch)
        writer.add_scalar('loss/loss_simple', np.mean(loss_simple), epoch)
        writer.add_scalar('loss/train_loss', np.mean(loss_task) + np.mean(loss_simple) , epoch)

        with torch.no_grad():
            loss = test(PointAE.eval(),sampler, valDataLoader)
            print('Test loss', loss.item())
            writer.add_scalar('loss/test_ae', loss, epoch)

            if (loss <= best):
                best = loss
                best_epoch = epoch + 1

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_samplenet.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'loss': loss,
                    'model_state_dict': sampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    writer.close()
    return best

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(123)
    from src.pctransforms import OnUnitCube, PointcloudToTensor
    from src.samplenet import SampleNet

    loss = []
    if args.test == 0:
        nums = [8,16,32,64]
        for num in nums:
            args.num_out_points = num
            args.log_dir = "samplenet_m" + str(num)
            args.sess = "samplenet_out{}".format(args.num_out_points)
            print(args)
            res = main(args)
            loss.append(res)
            print(loss)
        # b = [i for i in range(len(nums))]
        # plt.subplot(1, 1, 1)
        # plt.plot(b, acc, '-or')
        # # plt.legend(['Texas'])
        # plt.ylabel('Accuracy', fontsize=15, labelpad=15)
        # plt.xlabel('Number of points', fontsize=15, labelpad=15)
        # plt.savefig("./lstm.png", format="png", dpi=300)
        print(loss)
    else:
        nums = [8,16]
        for num in nums:
            args.num_out_points = num
            args.log_dir = "samplenet_m" + str(num)
            res = eval(args)
            loss.append(res)
            print(loss)
        # print(loss)
