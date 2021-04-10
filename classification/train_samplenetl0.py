"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
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
import shutil
import time
from time import localtime
from torch.utils.tensorboard import SummaryWriter

from data.modelnet_loader_torch import ModelNetCls
from src.pctransforms import OnUnitCube, PointcloudToTensor
import torchvision
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--sess', type=str, default="default", help='session')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')

    parser.add_argument('--sampler', required=True, default="samplenet", choices=['fps', 'samplenet', 'random', 'none'], type=str,
                        help='Sampling method.')
    parser.add_argument('--train-samplenet', action='store_true', default=True,help='Allow SampleNet training.')
    parser.add_argument('--train_cls', action='store_true', default=False, help='Allow calssifier training.')
    parser.add_argument('--num_out_points', type=int, default=32, help='sampled Point Number [default: 64]')
    parser.add_argument('--projection_group_size', type=int, default=8, help='projection_group_size')
    parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck_size')
    parser.add_argument('--a', default=0.01, type=float, help='alpha')
    parser.add_argument('--b', default=0.01, type=float, help='lmbda')
    parser.add_argument('--l0', default=1e-6, type=float, help='l0')
    parser.add_argument('--datafolder',  type=str, help='dataset folder')
    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
    # For testing
    parser.add_argument('--test', action='store_true',
                        help='Perform testing routine. Otherwise, the script will train.')
    return parser.parse_args()

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def get_datasets(args):
    transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])

    if not args.test:
        traindata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=True,
            download=True,
            folder=args.datafolder,
        )
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            folder=args.datafolder,
        )

        trainset = traindata
        testset = testdata
    else:
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            cinfo=None,
            folder=args.datafolder,
            include_shapes=True,
        )
        trainset = None
        testset = testdata

    return trainset, testset

def test(model,sampler, loader,criterion, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        sampler = sampler.eval()
        p0_simplified, p0_projected = sampler(points)
        # y = batched_index_select(points[:, :, 3:], 1, ind)
        # points = torch.cat((p0_projected, y), dim=-1)

        points = p0_projected.transpose(2, 1)

        classifier = model.eval()
        pred, trans_feat = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc,loss



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''


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
    args = parse_args()
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
    args.datafolder = 'modelnet40_ply_hdf5_2048'

    trainset, testset = get_datasets(args)
    # dataloader
    testDataLoader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    trainDataLoader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )


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

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = 0 #checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.train_cls:
        classifier.requires_grad_(True)
        classifier.train()
    else:
        classifier.requires_grad_(False)
        classifier.eval()

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

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            sampler.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []
    loss_task = []
    loss_l0 = []
    loss_simple = []

    def compute_samplenet_loss(sampler, data, epoch):
        """Sample point clouds using SampleNet and compute sampling associated losses."""

        p0 = data
        p0_simplified, p0_projected = sampler(p0)

        # Sampling loss
        p0_simplification_loss = sampler.get_simplification_loss(
            p0, p0_simplified, 64, 1, 0
        )

        simplification_loss = p0_simplification_loss
        sampled_data = (p0, p0_projected)
        # Projection loss
        #projection_loss = sampler.get_projection_loss()

        samplenet_loss = args.a * simplification_loss# + args.b * projection_loss

        samplenet_loss_info = {
            "simplification_loss": simplification_loss,
            "projection_loss": 0, #projection_loss,
        }

        return samplenet_loss, sampled_data, samplenet_loss_info

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        a = []
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]


            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            sampler = sampler.train()
            sampler_loss,sampled_data,sampler_loss_info= compute_samplenet_loss(sampler, points, epoch)
            a.append(sampler.m.item())
            # classifier = classifier.train()
            # points = torch.cat((sampled_data[1],points[:,:,3:]),dim=-1)
            points = sampled_data[1].transpose(2, 1)
            pred, trans_feat = classifier(points)
            loss_t = criterion(pred, target.long(), trans_feat)
            loss_l = sampler.loss * args.l0
            loss_s = sampler_loss
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss_task.append(loss_t.item())
            loss_l0.append(loss_l.item())
            loss_simple.append(loss_s.item())
            (loss_t + loss_l + loss_s).backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        writer.add_scalar('acc/train', train_instance_acc, epoch)
        writer.add_scalar('loss/loss_task', np.mean(loss_task), epoch)
        writer.add_scalar('loss/loss_l0', np.mean(loss_l0), epoch)
        writer.add_scalar('loss/loss_simple', np.mean(loss_simple), epoch)

        writer.add_scalar('number', np.array(a).mean(), epoch)
        sampler.k = torch.tensor(np.array(a).mean()).int()
        with torch.no_grad():
            instance_acc, class_acc,loss = test(classifier.eval(),sampler, testDataLoader,criterion)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            writer.add_scalar('acc/test_i', instance_acc, epoch)
            writer.add_scalar('acc/test_c', class_acc, epoch)
            writer.add_scalar('loss/test', loss, epoch)

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_samplenet.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from src import ChamferDistance, FPSSampler, RandomSampler
    from src import sputils
    from src.pctransforms import OnUnitCube, PointcloudToTensor
    from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg
    from src.samplenetl0 import SampleNet
    main(args)
