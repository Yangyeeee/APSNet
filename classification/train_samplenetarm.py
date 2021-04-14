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
    parser.add_argument('-b','--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--sess', type=str, default="default", help='session')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='decay rate [default: 0.7]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--ar', action='store_true', default=False, help='ar [default: False]')
    parser.add_argument('--sampler', required=True, default="samplenet", choices=['fps', 'samplenet', 'random', 'none'], type=str, help='Sampling method.')
    parser.add_argument('--train-samplenet', action='store_true', default=True,help='Allow SampleNet training.')
    parser.add_argument('--train_cls', action='store_true', default=False, help='Allow calssifier training.')
    parser.add_argument('--num_out_points', type=int, default=32, help='sampled Point Number [default: 32]')
    parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck_size')
    parser.add_argument('--k', default=1.0, type=float, help='k for sigmoid function')
    parser.add_argument('--bias', default=0.0, type=float, help='bias term to activate % points')
    parser.add_argument('--beta', default=1.0, type=float, help='beta for coverage loss')
    parser.add_argument('--l0', default=1.0, type=float, help='lambda for l0')
    parser.add_argument('--datafolder',  type=str, help='dataset folder')
    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
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

def test(model, sampler, loader, criterion, num_class=40):
    total_correct = 0.0
    total_samples = 0.0
    total_points = 0.0
    loss_task = []
    class_acc = np.zeros((num_class, 3))

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        sampler = sampler.eval()
        simplified = sampler(points, 0)
        # y = batched_index_select(points[:, :, 3:], 1, ind)
        # points = torch.cat((p0_projected, y), dim=-1)

        simplified = simplified.transpose(2, 1)

        classifier = model.eval()
        pred, trans_feat = classifier(simplified)
        loss = criterion(pred, target.long(), trans_feat)
        loss_task.append(loss.item())
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_samples += sampler.num.item() * points.size()[0]
        total_points += points.size()[0]
    class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = total_correct / total_points
    mean_loss_task = np.mean(loss_task)
    mean_samples = total_samples / total_points
    return instance_acc, class_acc, mean_loss_task, mean_samples



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
            k=args.k,
            bias=args.bias,
            bottleneck_size=args.bottleneck_size,
            input_shape="bnc",
            output_shape="bnc"
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
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.decay_rate)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    total_correct = 0.0
    loss_task = []
    loss_l0 = []
    loss_coverage = []

    def compute_samplenet_loss(sampler, data, epoch):
        """Sample point clouds using SampleNet and compute sampling associated losses."""

        simplified = sampler(data, epoch)

        # Sampling loss
        coverage_loss = sampler.get_simplification_loss(data, simplified)

        return coverage_loss, simplified

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        sampler = sampler.train()
        total_samples = 0.0

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)): #, smoothing=0.9):
        #for i in range(0):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]
            writer.add_scalar('loss/K', sampler.k1, epoch * len(trainDataLoader) + batch_id)

            points, target = points.cuda(), target.cuda()

            if sampler.training:
                sampler.forward_mode = True
                optimizer.zero_grad()

                coverage_loss, sampled_data = compute_samplenet_loss(sampler, points, epoch)
                total_samples += sampler.num.item() * points.size()[0]

                sampled_points = sampled_data.transpose(2, 1)
                pred, trans_feat = classifier(sampled_points)
                loss_t = criterion(pred, target.long(), trans_feat)
                loss_l = sampler.l0_loss * args.l0
                loss_c = coverage_loss * args.beta
                grad_l0 = sampler.l0_grad * args.l0

                sampler.eval()
                if args.ar is not True:
                    sampler.forward_mode = False
                    if sampler is not None: # and model.sampler.name == "samplenet":
                        points = points.transpose(2, 1)
                        sampler_loss1, sampled_data1 = compute_samplenet_loss(sampler, points, epoch)


                    # elif model.sampler is not None and model.sampler.name == "fps":
                    #     sampled_data = self.non_learned_sampling(model, data, device)
                    #     simplification_loss = torch.tensor(0, dtype=torch.float32)
                    #     projection_loss = torch.tensor(0, dtype=torch.float32)
                    #     sampler_loss = torch.tensor(0, dtype=torch.float32)
                    # else:
                    #     sampled_data = data
                    #     simplification_loss = torch.tensor(0, dtype=torch.float32)
                    #     projection_loss = torch.tensor(0, dtype=torch.float32)
                    #     sampler_loss = torch.tensor(0, dtype=torch.float32)

                    points = sampled_data1.transpose(2, 1)
                    pred, trans_feat = classifier(points)
                    loss1 = criterion(pred, target.long(), trans_feat)
                else:
                    loss1 = 0
                    sampler_loss1 = 0
                sampler.f1 = loss1 + sampler_loss1
                sampler.f2 = loss_t + loss_c

                sampler.train()

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                total_correct += correct.item()

                if sampler.training:
                    # model.sampler.loga.register_hook(model.sampler.update_phi_gradient)
                    grad_arm = sampler.update_phi_gradient()
                    sampler.loga.backward(gradient=(grad_arm + grad_l0), retain_graph=False)

                optimizer.step()
                global_step += 1
                loss_task.append(loss_t.item())
                loss_l0.append(loss_l.item())
                loss_coverage.append(loss_c.item())
                #if epoch >= 200:
                #    k1 = sampler.k1
                #    k1 +=  (sampler.loss - 0.03125)
                #    k1 = torch.max(torch.tensor([1e-5, k1])).item()
                #    k1 = torch.min(torch.tensor([1e5, k1])).item()
                #    sampler.k1 = k1

        scheduler.step()
        train_instance_acc = total_correct / len(trainset)
        mean_samples = total_samples / len(trainset)
        log_string('Train Instance Accuracy: %f, Mean samples: %f' % (train_instance_acc, mean_samples))
        writer.add_scalar('acc/train', train_instance_acc, epoch)
        writer.add_scalar('loss/loss_task', np.mean(loss_task), epoch)
        writer.add_scalar('loss/loss_l0', np.mean(loss_l0), epoch)
        writer.add_scalar('loss/loss_coverage', np.mean(loss_coverage), epoch)
        writer.add_scalar('loss/train_loss', np.mean(loss_task) + np.mean(loss_coverage) + np.mean(loss_l0), epoch)
        writer.add_scalar('number/train', mean_samples, epoch)

        #t_tmp= torch.tensor(np.array(a).mean()).int()
        #sampler.k = t_tmp
        with torch.no_grad():
            instance_acc, class_acc, mean_loss_task, mean_samples = test(classifier.eval(), sampler, testDataLoader, criterion)
            #sampler.k = 32
            #instance_acc32, class_acc32, _ = test(classifier.eval(), sampler, testDataLoader, criterion)
            #sampler.k = t_tmp
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f, Mean samples: %f'% (instance_acc, class_acc, mean_samples))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            writer.add_scalar('acc/test_i', instance_acc, epoch)
            writer.add_scalar('acc/test_c', class_acc, epoch)
            #writer.add_scalar('acc/test_i_32', instance_acc32, epoch)
            #writer.add_scalar('acc/test_c_32', class_acc32, epoch)
            writer.add_scalar('loss/test_task', mean_loss_task, epoch)
            writer.add_scalar('number/test', mean_samples, epoch)

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
    from src.pctransforms import OnUnitCube, PointcloudToTensor
    from src.samplenetl0arm import SampleNet
    main(args)
