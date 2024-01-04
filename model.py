import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer


class Net(nn.Module):
    def __init__(self, hs_band, ms_band, mid=64):
        super(Net, self).__init__()
        self.intro1 = nn.Sequential(
            nn.Conv2d(in_channels=hs_band, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )
        self.intro2 = nn.Sequential(
            nn.Conv2d(in_channels=ms_band, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )

        self.En1 = transformer.Cross_block(in_c1=mid, in_c2=mid, mid_c=64, cro=True)
        self.En2 = transformer.Cross_block(in_c1=64, in_c2=64, mid_c=96, cro=True)
        self.En3 = transformer.Cross_block(in_c1=96, in_c2=96, mid_c=128, cro=True)
        self.En4 = transformer.Cross_block(in_c1=128, in_c2=128, mid_c=256, cro=False)

        self.mid = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        self.cross1 = transformer.Block(in_c1=64, in_c2=64, out_c=64)
        self.cross2 = transformer.Block(in_c1=96, in_c2=96, out_c=96)
        self.cross3 = transformer.Block(in_c1=128, in_c2=128, out_c=128)

        self.De1 = transformer.UpDe(in_c=128, out_c=128)
        self.De2 = transformer.UpDe(in_c=128, out_c=96)
        self.De3 = transformer.UpDe(in_c=96, out_c=64)
        self.tail = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(64, hs_band, kernel_size=(1, 1))
        )
        # self.up1 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        #     # nn.LeakyReLU(),
        #     # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        # )
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        #     # nn.LeakyReLU(),
        #     # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        # )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        )

    def forward(self, y, z):
        # Y0 = y
        Y0 = F.interpolate(y, scale_factor=4, mode='bicubic', align_corners=False)
        Y = self.intro1(Y0)
        Z = self.intro2(z)
        # print('--------down--------')
        c1, y1, z1 = self.En1(Y, Z)
        c2, y2, z2 = self.En2(y1, z1)
        c3, y3, z3 = self.En3(y2, z2)
        y4, z4 = self.En4(y3, z3)

        mid = self.mid(torch.cat((y4, z4), dim=1))

        # print('--------cross--------')
        c1 = self.cross1(c1, c1)
        c2 = self.cross2(c2, c2)
        c3 = self.cross3(c3, c3)

        # print('-------up------')
        x1 = self.De1(c3, mid)
        x2 = self.De2(c2, x1)
        x3 = self.De3(c1, x2)

        # print('-----end----')
        # # # #case 0:
        # out = self.conv(torch.cat((x3, Y0, z), dim=1))
        # out = self.tail(out) + Y0
        # # # #case 1:
        # out = torch.cat((self.up1(x1), self.up2(x2), x3), dim=1)
        # out = self.tail(out) + Y0
        # # # #case 2:
        out = self.conv(torch.cat((self.up3(x1), x2), dim=1))
        out = self.tail(torch.cat((self.up4(out), x3), dim=1)) + Y0
        return out


if __name__ == '__main__':
    y = torch.randn(1, 31, 16, 16)
    z = torch.randn(1, 3, 64, 64)
    net = Net(31, 3)
    out = net(y, z)
