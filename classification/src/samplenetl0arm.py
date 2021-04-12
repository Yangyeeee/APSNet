from __future__ import print_function

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN


try:
    from .soft_projection import SoftProjection
    from .chamfer_distance import ChamferDistance
    from . import sputils
    # from . import soft
except (ModuleNotFoundError, ImportError) as err:
    print(err.__repr__())
    from soft_projection import SoftProjection
    from chamfer_distance import ChamferDistance
    import sputils
    # import soft


num_iter = int(1e2)
epsilon=5e-2
sig = nn.Sigmoid()
hardtanh = nn.Hardtanh(0,1)
gamma = -0.1
zeta = 1.1
beta = 0.66
eps = 1e-20
const1 = beta*np.log(-gamma/zeta + eps)

def l0_train(logAlpha, min, max):
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def l0_test(logAlpha, min, max):
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask

def get_loss2(logAlpha):
    return sig(logAlpha - const1)



def axis_to_dim(axis):
    """Translate Tensorflow 'axis' to corresponding PyTorch 'dim'"""
    return {0: 0, 1: 2, 2: 3, 3: 1}.get(axis)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class SampleNet(nn.Module):
    def __init__(
        self,
        bottleneck_size,
        input_shape="bcn",
        output_shape="bcn"
    ):
        super().__init__()
        self.name = "samplenet"
        # self.sampler = soft.TopK_custom(num_out_points, epsilon=epsilon, max_iter=num_iter)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1024)
        # self.fc1 = torch.nn.Conv1d(bottleneck_size*2, 256, 1)
        # self.fc2 = torch.nn.Conv1d(256, 256, 1)
        # self.fc3 = torch.nn.Conv1d(256, 256, 1)
        # self.fc4 = torch.nn.Conv1d(256, 1, 1)
        # self.fcp = nn.Linear(bottleneck_size, 1, bias=False)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError("allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l0_loss = torch.tensor(0)
        self.l0_grad = torch.tensor(0)
        self.num = torch.tensor(0)
        self.m = None
        self.loga = torch.tensor(0)
        self.f1 = torch.tensor(0)
        self.f2 = torch.tensor(0)
        #self.k = 1024
        self.ind = 0
        self.hardsigmoid = False
        self.k1 = 1
        self.u = None
        self.local_rep = True
        self.forward_mode = True
        self.ar = True
        self.fc4.bias.data.fill_(0. / self.k1)

    def sample_z(self,loga):

        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1) #.detach()
        else:
            pi = torch.sigmoid(self.k1 * loga) #.detach()

        if self.forward_mode:
            z = torch.zeros_like(loga)
            self.u = torch.zeros(loga.shape[1]).to(loga.device).uniform_(0, 1).expand(loga.shape[0], loga.shape[1])  # torch.zeros_like(loga).uniform_(0, 1)
            z[self.u < pi] = 1

            #if self.training:
            #    self.u = torch.zeros(loga.shape[1]).to(loga.device).uniform_(0, 1).expand(loga.shape[0], loga.shape[1]) #torch.zeros_like(loga).uniform_(0, 1)
            #    z[self.u < pi] = 1
            #    self.train_z = z
            #else:
            #    z[loga > 0] = 1
            #    self.test_z = z
        else:
            pi2 = 1 - pi
            if self.u is None:
                raise Exception('Forward pass first')
            z = torch.zeros_like(loga)
            z[self.u > pi2] = 1

        return z

    def get_loss(self, loga):
        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(self.k1 * loga)

        l0 = pi.mean()
        return l0

    def get_grad_loss(self, loga):
        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(self.k1 * loga)

        l0 = pi.mean()
        grad = pi*(1-pi)*self.k1/loga.numel() #shape[0]
        return grad, l0

    def update_phi_gradient(self):
        # only deal with first part of gradient
        # regularization part will be handled by pytorch
        k = self.k1
        if self.ar:
            e = k * (self.f2 * (1 - 2 * self.u))
        else:
            e = k * ((self.f1 - self.f2) * (self.u - .5))
        return e

    def forward(self, x: torch.Tensor,epoch):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        a = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints
        # Max pooling for global feature vector:
        y = torch.max(a, 2)[0]

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        loga = self.fc4(y) #+ self.bias_l0).squeeze()

        #y = F.relu(self.fc1(y))
        #y = F.relu(self.fc2(y))
        #y = F.relu(self.fc3(y))
        #loga = self.fc4(y) #+ self.bias_l0 #.reshape((x.shape[0],x.shape[2]))

        self.loga = loga

        if self.training:
            self.l0_grad, self.l0_loss = self.get_grad_loss(loga)

        m = self.sample_z(loga)
        self.num = (m > 0).sum(-1).float().mean()
        y = x * m.unsqueeze(1)

        #ind = torch.topk(loga, self.k, dim=-1)[1]
        #self.ind  = ind
        # Simplified points
        #simp = batched_index_select(y, 2, ind)
        simp = y

        # Change to output shapes
        if self.output_shape == "bnc":
            simp = simp.permute(0, 2, 1)

        # Assert contiguous tensors
        simp = simp.contiguous()

        return simp

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc):
        if not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matricesself.skip_projection or
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        #max_cost = torch.max(cost_p1_p2, dim=1)[0]
        #max_cost = torch.mean(max_cost)
        #cost_p1_p2 = torch.mean(cost_p1_p2)
        loss = torch.mean(cost_p2_p1)
        #loss = cost_p2_p1
        return loss

    def get_simplification_loss1(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        if not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        if self.skip_projection or not self.training:
            return torch.tensor(0).to(sigma)
        return sigma


if __name__ == "__main__":
    point_cloud = np.random.randn(1, 3, 1024)
    point_cloud_pl = torch.tensor(point_cloud, dtype=torch.float32).cuda()
    net = SampleNet(5, 128, group_size=10, initial_temperature=0.1, complete_fps=True)

    net.cuda()
    net.eval()

    for param in net.named_modules():
        print(param)

    simp, proj, match = net.forward(point_cloud_pl)
    simp = simp.detach().cpu().numpy()
    proj = proj.detach().cpu().numpy()
    match = match.detach().cpu().numpy()

    print("*** SIMPLIFIED POINTS ***")
    print(simp)
    print("*** PROJECTED POINTS ***")
    print(proj)
    print("*** MATCHED POINTS ***")
    print(match)

    mse_points = np.sum((proj - match) ** 2, axis=1)
    print("projected points vs. matched points error per point:")
    print(mse_points)
