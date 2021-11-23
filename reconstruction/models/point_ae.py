import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet import Encoder, Decoder


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


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.encoder = Encoder( channel=channel)
        self.decoder = Decoder()

    def forward(self, x):
        feat = self.encoder(x)
        recon = self.decoder(feat)
        return recon

    def get_loss(self, x, recon):
        cost_p2_p1 = square_distance(x, recon).min(-1)[0]
        cost_p1_p2 = square_distance(recon, x).min(-1)[0]
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + cost_p2_p1
        return loss

    def get_per_loss(self, x, recon):
        cost_p2_p1 = square_distance(x, recon).min(-1)[0]
        cost_p1_p2 = square_distance(recon, x).min(-1)[0]
        loss = cost_p1_p2 + cost_p2_p1
        return loss.mean(-1)


