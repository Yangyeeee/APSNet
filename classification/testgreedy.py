import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import time
import provider
from time import localtime
from torch.utils.tensorboard import SummaryWriter
from data.modelnet_loader_torch import ModelNetCls
import torchvision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('SSN')
    parser.add_argument('-b','--batch_size', type=int, default=2468, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--log_dir', type=str, default='pointnet', help='experiment root')
    parser.add_argument('--sess', type=str, default="default", help='session')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--datafolder',  type=str, help='dataset folder')
    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
    parser.add_argument('--beta', default=1.0, type=float, help='beta for coverage loss')
    parser.add_argument('--max', action='store_true', default=False, help='using max')
    parser.add_argument('--fps', action='store_true', default=False, help='using FPS')
    # For testing
    parser.add_argument('--test', action='store_true', help='Perform testing routine. Otherwise, the script will train.')
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

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    dist += torch.sum(src ** 2, 1).view(B, N, 1)
    dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)      # Bx1024
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def test_fps(model, loader, writer, num_class=40):

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]

        points, target = points.cuda(), target.cuda()
        p_ind = farthest_point_sample(points,32)
        p = index_points(points, p_ind)
        for i in range(32):
            class_acc = np.zeros((num_class, 3))
            simplified = p[:,0:i+1]
            classifier = model.eval()
            simplified = simplified.transpose(2, 1)
            pred, trans_feat = classifier(simplified)
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(p[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()

            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = np.mean(class_acc[:, 2])
            instance_acc = correct.item() / p.size()[0]
            print('%f  Test Instance Accuracy: %f, Class Accuracy: %f' % (i, instance_acc, class_acc))
            writer.add_scalar('acc/test_i', instance_acc, i)
            writer.add_scalar('acc/test_c', class_acc, i)


def test_greedy_batch(model, loader, writer, num_class=40, fps=False):

    unselected = torch.ones((2468, 1024)).bool().cuda()
    selected = torch.zeros((2468, 1024)).bool().cuda()
    index = (torch.ones((2468, 1024)) * (torch.tensor([i for i in range(1024)]).view(-1, 1024))).long().cuda()
    classifier = model.eval()

    for i in range(32):
        class_acc = np.zeros((num_class, 3))
        p = []
        t = []
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            points, target = data
            points = points.transpose(2, 1)
            target = target[:, 0]
            N = points.shape[0]
            points, target = points.cuda(), target.cuda()
            p.append(points)
            t.append(target)
            ind_l = j*args.batch_size
            ind_h = min((j+1)*args.batch_size,2468)
            selected_batch = selected[ind_l:ind_h]
            unselected_batch = unselected[ind_l:ind_h]
            selected_ind_batch = index[ind_l:ind_h][selected_batch].reshape(N, -1)
            unselected_ind_batch = index[ind_l:ind_h][unselected_batch].reshape(N, -1)
            selected_point = batched_index_select(points, 2, selected_ind_batch)
            unselected_point = batched_index_select(points, 2, unselected_ind_batch)
            logit = []

            if fps:
                for k in range(1024 - i):
                    tmp_index = unselected_ind_batch[:,k].reshape(N,-1)
                    tmp = batched_index_select(points, 2, tmp_index)
                    if selected_point.shape[2] != 0:
                        simplified = torch.cat((selected_point, tmp), dim=2)
                    else:
                        simplified = tmp
                    pred, trans_feat = classifier(simplified)
                    tmp_logit = pred.max(-1, keepdim=True)[0]
                    logit.append(tmp_logit)
                score = torch.cat(logit, dim=-1)

                if i == 0:
                    #sel = torch.gather(unselected_ind_batch, dim=-1, index=torch.LongTensor(N, 1).random_(0, 1024).cuda())
                    best_idx = score.max(-1, keepdim=True)[1]
                else:
                    cost_p2_p1 = square_distance(unselected_point, selected_point).min(-1)[0]
                    best_idx = (args.beta * cost_p2_p1 + (1 - args.beta) * score).max(-1, keepdim=True)[1]
                sel = torch.gather(unselected_ind_batch, dim=-1, index=best_idx)
            else:
                for k in range(1024 - i):
                    tmp_index = unselected_ind_batch[:,k].reshape(N,-1)
                    tmp = batched_index_select(points, 2, tmp_index)
                    if selected_point.shape[2] != 0:
                        simplified = torch.cat((selected_point, tmp), dim=2)
                    else:
                        simplified = tmp
                    pred, trans_feat = classifier(simplified)

                    cost_p2_p1 = square_distance(unselected_point, simplified).min(-1)[0]
                    if args.max:
                        coverage = -1 * torch.max(cost_p2_p1, dim=-1, keepdim=True)[0]
                    else:
                        coverage = -1 * torch.mean(cost_p2_p1, dim=-1, keepdim=True)

                    tmp_logit =  args.beta * coverage + (1 - args.beta) * pred.max(-1,keepdim=True)[0]
                    logit.append(tmp_logit)

                sel = torch.gather(unselected_ind_batch, dim=-1, index=torch.argmax(torch.cat(logit, dim=-1), dim=-1, keepdim=True))
                #sel = torch.gather(unselected_ind, dim=-1, index=torch.LongTensor(2468, 1).random_(0, 1024-i).cuda())

            selected_ind = torch.cat((selected_ind_batch, sel), dim=-1)
            unselected[ind_l: ind_h] = torch.ones((N, 1024),dtype=bool).cuda().scatter_(dim=-1,index=selected_ind,src=torch.tensor(0,dtype=bool))
            selected[ind_l: ind_h] =  torch.zeros((N, 1024),dtype=bool).cuda().scatter_(dim=-1,index=selected_ind,src=torch.tensor(1,dtype=bool))

        selected_ind = index[selected].reshape(2468, -1)
        p = torch.cat(p,dim=0)
        t = torch.cat(t,dim=0)
        simplified = batched_index_select(p, 2, selected_ind)
        classifier = model.eval()
        pred, trans_feat = classifier(simplified)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(t.cpu()):
            classacc = pred_choice[t==cat].eq(t[t ==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(p[t ==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(t.long().data).cpu().sum()

        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc = np.mean(class_acc[:,2])
        instance_acc = correct.item()/ p.size()[0]
        print('%f  Test Instance Accuracy: %f, Class Accuracy: %f' % ( i, instance_acc, class_acc))
        writer.add_scalar('acc/test_i', instance_acc,i )
        writer.add_scalar('acc/test_c', class_acc, i)



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
    testDataLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40
    MODEL = importlib.import_module(args.model)


    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')



    with torch.no_grad():

        test_greedy_batch(classifier.eval(), testDataLoader, writer, fps=args.fps)
        # test_fps(classifier.eval(), testDataLoader, writer)

    writer.close()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from src.pctransforms import OnUnitCube, PointcloudToTensor
    main(args)
