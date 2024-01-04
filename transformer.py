import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class ffn(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn1(x)


class Cross_block(nn.Module):

    def __init__(self, in_c1, in_c2, mid_c, cro) -> None:
        super().__init__()
        self.cro = cro
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c1, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c2, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
        )
        self.act = SCAM(mid_c)

    def forward(self, Y, Z):
        # print(Y.shape, Z.shape)
        Y1 = self.conv1(Y)
        Z1 = self.conv2(Z)
        if self.cro:
            c = self.act(Y1, Z1)
            Y1 = self.down1(Y1)
            Z1 = self.down1(Z1)
            return c, Y1, Z1
        Y1 = self.down1(Y1)
        Z1 = self.down2(Z1)
        return Y1, Z1


class UpDe(nn.Module):

    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(2, 2), stride=(2, 2)),
            # nn.Conv2d(in_c, in_c * 4, kernel_size=(1, 1), stride=(1, 1), padding=0),
            # nn.PixelShuffle(2),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Conv2d(in_channels=out_c + out_c, out_channels=out_c, kernel_size=(3, 3), padding=1)
        # self.conv = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1)

    def forward(self, infr, latt):
        x = self.up(latt)
        # with
        out = torch.cat((x, infr), dim=1)
        out = self.conv(out)
        # without
        # out = self.conv(x)
        return out


class SCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = nn.LayerNorm(c)
        self.norm_r = nn.LayerNorm(c)
        self.norm = nn.LayerNorm(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        out = F_r2l + F_l2r
        return x_l + out + x_r


class Block(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super().__init__()
        self.Embedding1 = nn.Sequential(
            nn.Linear(in_c1, out_c),
            nn.LayerNorm(out_c)
        )
        # self.patch = nn.Conv2d(in_channels=in_c1, out_channels=in_c1, kernel_size=8, stride=8)
        self.Embedding2 = nn.Sequential(
            nn.Linear(in_c2, out_c),
            nn.LayerNorm(out_c)
        )
        self.attn = DualT(dim=out_c, heads=4, dim_head=16)
        self.FFN = ffn(out_c, out_c)

    def forward(self, X, Y):
        H = X.size(2)
        E1 = rearrange(X, 'B c H W -> B (H W) c', H=H)
        E1 = self.Embedding1(E1)
        E2 = rearrange(Y, 'B c H W -> B (H W) c', H=H)
        E2 = self.Embedding2(E2)
        attn = self.attn(E1, E2)
        out = self.FFN(attn) + attn
        out = rearrange(out, 'B (H W) C -> B C H W', H=H)
        return out


class DualT(nn.Module):

    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv_1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_qkv_3 = nn.Linear(dim * 2, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        # print(x.shape, y.shape)
        b, n, c, h = x.shape[0], x.shape[1], x.shape[2], self.heads
        qkv1 = self.to_qkv_1(x).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv1)
        qkv2 = self.to_qkv_2(y).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=h), qkv2)

        dots1 = torch.einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)
        out2 = torch.einsum('b h i j, b h j d -> b h i d', attn2, v2)
        out2 = rearrange(out2, 'b h d n -> b n (h d)')

        input = torch.add(out1, out2)
        output = self.norm(self.to_out(input))
        # output = output + x + y
        return output


if __name__ == '__main__':
    y = torch.randn(1, 32, 64, 64)
    z = torch.randn(1, 32, 64, 64)
    net = Cross_block(in_c1=32, in_c2=32, mid_c=64, cro=True)
    net(y, z)
