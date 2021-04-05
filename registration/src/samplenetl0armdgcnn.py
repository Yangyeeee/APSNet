from __future__ import print_function

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from .model import  get_graph_feature,knn


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
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"
        # self.sampler = soft.TopK_custom(num_out_points, epsilon=epsilon, max_iter=num_iter)
        self.kk = 8

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, bottleneck_size, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(bottleneck_size * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 1024)


        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = torch.tensor(0)
        self.grad = torch.tensor(0)
        self.num = torch.tensor(0)
        self.m = None
        # self.a = [1024,512,256,128,64]
        self.bias_l0 = nn.Parameter(torch.FloatTensor([5]))
        self.loga = torch.tensor(0)
        self.f1 = torch.tensor(0)
        self.f2 = torch.tensor(0)
        self.k = 1024
        self.ind = 0
        self.hardsigmoid = False
        self.k1 = 1
        self.u = None
        self.local_rep = True
        self.forward_mode = True
        self.ar = True

    def sample_z(self,loga):

        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1)#.detach()
        else:
            pi = torch.sigmoid(self.k1 * loga)#.detach()
        if self.forward_mode:
            z = torch.zeros_like(loga)
            if self.training:
                # if self.local_rep:
                #     self.u = torch.FloatTensor(1024).uniform_(0, 1).expand(batch_size, self.in_features).to(self.z_phi.device)
                # else:
                self.u = torch.zeros(loga.shape[1]).to(loga.device).uniform_(0, 1).expand(loga.shape[0], loga.shape[1]) #torch.zeros_like(loga).uniform_(0, 1)

                z[self.u < pi] = 1
                # if self.use_t_in_training:
                #     z[pi.expand(batch_size, self.in_features) < opt.t] = 0
                self.train_z = z
            else:
                z[loga > 0] = 1

                # if opt.use_t_in_testing:
                #     z = pi.expand(batch_size, self.in_features)
                #     z[z < opt.t] = 0
                self.test_z = z
        else:
            pi2 = 1 - pi
            if self.u is None:
                raise Exception('Forward pass first')
            z = torch.zeros_like(loga)
            z[self.u > pi2] = 1
            # if opt.use_t_in_training:
            #     z[pi.expand(batch_size, self.in_features) < opt.t] = 0
        # z = (z - loga).detach() + loga
        return z

    def get_loss(self, loga):
        # return sig(logAlpha - const1)
        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(self.k1 * loga)

        l0 = pi.sum(-1).mean()
        return l0

    def get_grad(self, loga):
        # return sig(logAlpha - const1)
        if self.hardsigmoid:
            pi = F.hardtanh(self.k1 * loga / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(self.k1 * loga)

        grad = pi*(1-pi)*self.k1
        return grad
    # def update_phi_gradient(self, grad):
    #     # only deal with first part of gradient
    #     # regularization part will be handled by pytorch
    #     k = self.k1
    #     if self.ar:
    #         e = k * (self.f2 * (1 - 2 * self.u))#.mean(dim=0)
    #     else:
    #         e = k * ((self.f1 - self.f2) * (self.u - .5))#.mean(dim=0)
    #     return e.type_as(grad)
    def update_phi_gradient(self):
        # only deal with first part of gradient
        # regularization part will be handled by pytorch
        k = self.k1
        if self.ar:
            e = k * (self.f2 * (1 - 2 * self.u))#.mean(dim=0)
        else:
            e = k * ((self.f1 - self.f2) * (self.u - .5))#.mean(dim=0)
        return e

    def forward(self, x: torch.Tensor,epoch):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        batch_size = x.size(0)
        y = get_graph_feature(x, k=self.kk)
        y = self.conv1(y)
        y1 = y.max(dim=-1, keepdim=False)[0]

        y = get_graph_feature(y1, k=self.kk)
        y = self.conv2(y)
        y2 = y.max(dim=-1, keepdim=False)[0]

        y = get_graph_feature(y2, k=self.kk)
        y = self.conv3(y)
        y3 = y.max(dim=-1, keepdim=False)[0]

        y = get_graph_feature(y3, k=self.kk)
        y = self.conv4(y)
        y4 = y.max(dim=-1, keepdim=False)[0]

        y = torch.cat((y1, y2, y3, y4), dim=1)

        y = self.conv5(y)
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)
        y = torch.cat((y1, y2), 1)

        y = F.leaky_relu(self.bn6(self.linear1(y)), negative_slope=0.2)
        y = self.dp1(y)
        y = F.leaky_relu(self.bn7(self.linear2(y)), negative_slope=0.2)
        y = self.dp2(y)
        y = self.linear3(y)

        loga = (y + self.bias_l0).squeeze()

        # if self.training:
        #     m = l0_train(loga, 0, 1)
        # else:
        #     m = l0_test(loga, 0, 1)
        self.loga = loga

        if self.training:
            self.grad = self.get_grad(loga)
            self.loss = self.get_loss(loga)

        m = self.sample_z(loga)

        # # Set gradients w.r.t. y_hard gradients w.r.t. y
        # m = (m - loga).detach() + loga

        y = x * m.unsqueeze(1)

        # m = self.sampler(loga)
        # y = x * m.unsqueeze(1)
        ind = torch.topk(loga, self.k, dim=-1)[1]
        self.ind  = ind
        self.num = (m > 0).squeeze().sum(-1).float().mean()
        self.m = m.view(-1)
        #
        # Simplified points
        simp = batched_index_select(y, 2, ind)
        match = None
        proj = None

        # Projected points
        if self.training:
            if not self.skip_projection:
                proj = self.project(point_cloud=x, query_cloud=y)
            else:
                proj = simp

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            # ind = torch.topk(A, self.num_out_points,dim=-1)[1]
            # c = batched_index_select(y, 2, ind)
            # ind = torch.topk(dist.squeeze(), self.num_out_points, dim=-1)[1]

            y = batched_index_select(x, 2, ind)
            match = y.permute(0,2,1) #batched_index_select(y, 2, ind).permute(0,2,1)
            # _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())
            #
            # """Notice that we detach the tensors and do computations in numpy,
            # and then convert back to Tensors.
            # This should have no effect as the network is in eval() mode
            # and should require no gradients.
            # """
            #
            # # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            # x = x.permute(0, 2, 1).cpu().detach().numpy()
            # y = y.permute(0, 2, 1).cpu().detach().numpy()
            #
            # idx = idx.cpu().detach().numpy()
            # idx = np.squeeze(idx, axis=1)
            #
            # z = sputils.nn_matching(
            #     x, idx, self.num_out_points, complete_fps=self.complete_fps
            # )
            #
            # # Matched points are in B x N x 3 format.
            # match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        out = proj if self.training else match

        return simp, out

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        if not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matricesself.skip_projection or
        # samp_pc = batched_index_select(samp_pc, 1, self.ind)
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest pointcost_p1_p2 + max_cost + (gamma + delta * pc_size) *
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p2_p1
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
