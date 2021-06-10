from __future__ import print_function

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# def gumbel_softmax(logits, temp=0.3, k=32, train=False):
#     """
#     input: [*, n_class]
#     return: flatten --> [*, n_class] an one-hot vector
#     """
#
#     logits = logits.unsqueeze(1)
#     U = torch.rand((logits.shape[0],k,logits.shape[-1])).type_as(logits) + 1e-20
#     g = torch.log(U) - torch.log(1 - U)
#     noisy_logits  = logits + g
#     samples = F.softmax(noisy_logits  / temp, dim=-1)
#     return samples

def gumbel_softmax(logits, temp=0.3, k=32, train=False):
    tmp = []
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    for i in range(k):

        z = F.softmax(logits/temp,dim=-1)
        logits = logits + torch.log(1-z + 1e-8)
        tmp.append(z.unsqueeze(1))
    samples = torch.cat(tmp, dim=1)
    return samples


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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def _calc_distances(p0, points):
    return torch.pow((p0 - points), 2).sum(-1)


def fps_from_given_pc(pts, k, given_pc):
    farthest_pts = torch.zeros((k, 3)).to(given_pc.device)
    t = given_pc.shape[0]
    farthest_pts[0:t] = given_pc

    distances = square_distance(given_pc.unsqueeze(0), pts.unsqueeze(0)).min(1)[0].squeeze()
    for i in range(t, k):
        farthest_pts[i] = pts[torch.max(distances, dim=-1)[1]]
        dis = _calc_distances(farthest_pts[i], pts)
        distances = torch.where(distances < dis, distances, dis)
    return farthest_pts


def selecting(full_pc, m):
    num = (m > 0).sum(-1).max().item()
    batch_size = full_pc.shape[0]
    out_pc = torch.zeros((batch_size, num, 3)).to(m.device)

    for i in range(0, batch_size):
        cur_idx = torch.nonzero(m[i] != 0, as_tuple=False).squeeze()
        out_pc[i] = fps_from_given_pc(full_pc[i], num, full_pc[i][cur_idx])

    return out_pc


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
        self.bn6 = nn.BatchNorm1d(1)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1024)
        # self.fc1 = torch.nn.Conv1d(bottleneck_size, 256, 1)
        # self.fc2 = torch.nn.Conv1d(256, 256, 1)
        # self.fc3 = torch.nn.Conv1d(256, 256, 1)
        # self.fc4 = torch.nn.Conv1d(256, 3, 1)
        # self.fcp = nn.Linear(bottleneck_size, 1, bias=False)


        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # # projection and matching
        # self.project = SoftProjection(
        #     group_size, initial_temperature, is_temperature_trainable, min_sigma
        # )
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
        self.k = 32
        self.m = 0
        self.tmp = 0
        self.t = torch.tensor(1, dtype=torch.float32) #torch.nn.Parameter(torch.tensor(0.8,requires_grad=True, dtype=torch.float32))
        self.min_t = torch.tensor(0.3, dtype=torch.float32)
        self.loss = 0

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints
        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        loga = self.fc4(y)


        # Simplified points
        simp = None
        match = None
        proj = None

        if self.training:
            t = torch.max(self.t**2, self.min_t).to(y.device)
            m = gumbel_softmax(loga,temp=t, k=self.k,train=self.training)
            self.m = m
            self.tmp = t
            y = (x.unsqueeze(1) * m.unsqueeze(-2)).sum(-1)  # nx1x3x1024 * nx32x1x1024 --> nx32x3

            # Simplified points
            proj = y
            simp = y

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            # ind = torch.topk(A, self.num_out_points,dim=-1)[1]
            # c = batched_index_select(y, 2, ind)
            # ind = torch.topk(dist.squeeze(), self.num_out_points, dim=-1)[1]
            ind = torch.topk(loga, self.k, dim=-1)[1]
            y = batched_index_select(x, 2, ind)
            match = y.permute(0,2,1) #batched_index_select(y, 2, ind).permute(0,2,1)

        # # Change to output shapes

        # Assert contiguous tensors
        if proj is not None:
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
        if self.skip_projection or not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        # cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        cost_p2_p1 = square_distance(ref_pc, samp_pc).min(-1)[0]
        cost_p1_p2 = square_distance(samp_pc, ref_pc).min(-1)[0]
        max_cost = torch.max(cost_p1_p2, dim=1)[0]
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
