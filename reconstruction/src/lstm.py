from __future__ import print_function

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def fps_from_given_pc(pts, k, given_pc):
    farthest_pts = torch.zeros((given_pc.shape[0], k, 3)).to(given_pc.device)
    t = given_pc.shape[1]
    farthest_pts[:,0:t] = given_pc
    distances = square_distance(given_pc, pts).min(1)[0]
    for i in range(t, k):
        farthest_pts[:,i] = batched_index_select(pts,1,torch.max(distances, dim=-1)[1].reshape(-1,1)).squeeze()
        dis = square_distance(farthest_pts[:,i].unsqueeze(1), pts).min(1)[0]
        distances = torch.where(distances < dis, distances, dis)
    return farthest_pts


class apsnet(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        layer=2,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "apsnet"
        self.layer = layer
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)
        self.lstm = nn.LSTM(3, 128, self.layer, batch_first=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        # projection and matching
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
            warnings.warn("apsnet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape
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
        y1 = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x 1024

        y = torch.max(y1, 2)[0]  # Batch x 128

        # states = (torch.zeros(self.layer, x.shape[0], 128).to(x.device),
        #           torch.zeros(self.layer, x.shape[0], 128).to(x.device))
        states = (torch.zeros(self.layer, x.shape[0], 128).to(x.device),
                  y.unsqueeze(0).repeat(self.layer,1,1))
        inputs = torch.zeros( x.shape[0],1, 3).to(x.device)
        res = []
        if self.training:
            num_out_points = self.num_out_points
        else:
            num_out_points = min(self.num_out_points,128)
        for i in range(num_out_points):

            outputs, states = self.lstm(inputs, states)  #Bx1x128
            p = torch.bmm(outputs,y1)
            p = F.softmax(p,dim=-1)
            inputs = torch.bmm(x,p.permute(0,2,1))
            res.append(inputs)
            inputs = inputs.permute(0,2,1)

        # Simplified points
        simp = torch.cat(res,dim=-1)
        match = None
        proj = None

        # Projected points
        if self.training:
            proj = simp
        else:  # Inference
            num = 128
            if self.num_out_points > num:
                match = fps_from_given_pc(x.permute(0,2,1),self.num_out_points,simp.permute(0,2,1))
            else:
                match = simp.permute(0,2,1)

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

        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p2_p1 = square_distance(ref_pc, samp_pc).min(-1)[0]
        cost_p1_p2 = square_distance(samp_pc, ref_pc).min(-1)[0]
        max_cost = torch.max(cost_p1_p2, dim=1)[0]
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

