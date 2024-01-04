import torch
import torch.nn as nn
import DataSets as DS
import scipy.io as sio
import numpy as np
import model
import cv2
import argparse
import os
import torch.nn.functional as F
import torch
from thop import profile, clever_format
from utiils import PSNR_GPU, quality_mesure_fun, checkFile, merge_Cave_test, cutCAVEPieces_Test
import logging
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class CD_UNet:
    def __int__(self, args):
        pass

    @classmethod
    def load_train(self, path, bath_size):
        datasets = DS.Dataset(path)
        trainloader = torch.utils.data.DataLoader(datasets, batch_size=bath_size, shuffle=True)
        return trainloader

    @classmethod
    def get_R(self):
        R = np.array(
            [[2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
             [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
             [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        R = self.sumtoOne(R)
        R = torch.from_numpy(R.astype(np.float32))
        R = R.unsqueeze(-1)
        R = R.unsqueeze(-1)
        R = R.to(device)
        return R

    @classmethod
    def sumtoOne(self, R):
        div = np.sum(R, axis=1)
        div = np.expand_dims(div, axis=-1)
        R = R / div
        return R

    @classmethod
    def cpt_target(self, X):
        B = cv2.getGaussianKernel(args.ratio, 2) * cv2.getGaussianKernel(args.ratio, 2).T
        # B转为tensor
        B = np.reshape(B, newshape=(1, 1, args.ratio, args.ratio))
        B = torch.tensor(B, dtype=torch.float).to(device)
        B = B.repeat(args.hs_band, 1, 1, 1)
        pad = int(args.ratio - 1) // 2
        Y = F.conv2d(X, B, None, stride=(args.ratio, args.ratio), padding=(pad, pad), groups=args.hs_band)
        Z = F.conv2d(X, self.get_R(), None)
        # Z = torch.mean(X, dim=1)
        return Y, Z

    @classmethod
    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])
        fh = logging.FileHandler(filename, "w")
        logger.addHandler(fh)
        return logger

    @classmethod
    def train(self):
        ###################
        # train the model #
        ###################
        train_loader = self.load_train(args.train_path, args.batch_size)
        # valid_loader = self.load_train(args.valid_path, 8)
        Net = model.Net(args.hs_band, args.ms_band).to(device)
        log_path = './output.txt/'
        logger = self.get_logger(log_path)
        g_optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
        # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[100, 150, 180], gamma=0.8)
        Loss1 = nn.L1Loss()

        for epoch in range(args.max_epoch):
            avg_psnr = 0
            total_loss = 0
            for i, data1 in enumerate(train_loader, 0):
                Y = data1['Y'].to(device)
                Z = data1['Z'].to(device)
                label = data1['X'].to(device)
                g_optimizer.zero_grad()
                X = Net(Y, Z)
                loss = Loss1(X, label)  # + Loss1(X2, label)
                total_loss = total_loss + loss
                psnr = PSNR_GPU(label, X)
                avg_psnr = avg_psnr + psnr
                loss.backward()
                g_optimizer.step()
                lr = g_optimizer.state_dict()['param_groups'][0]['lr']

            avg_psnr = avg_psnr / len(train_loader)
            total_loss = total_loss / len(train_loader)
            print('Train:Epoch[%d/%d] lr:%.8f Avg_psnr:%.8f Loss:%.8f--------------------' % (
                (epoch + 1), args.max_epoch, lr, avg_psnr, total_loss))
            # logger.info('The %d epoch: learning rate = %f, Training_Loss = %.4f, PSNR = %.4f' % (
            #     epoch, lr, total_loss, avg_psnr))
            if (epoch + 1) % 10 == 0:
                checkFile(args.Net_save_path)
                torch.save(Net.state_dict(), args.Net_save_path + 'Netbase_r4_4_' + str(epoch + 1) + '.pkl')
                print('Models save to ./Models/Net.pkl')
            # lr_schedule.step()

    @classmethod  # non-overlap
    def Test(self):
        net = model.Net(args.hs_band, args.ms_band)
        # net = Nettt.gloab_net(args.hs_band, args.ms_band)
        net_dict = torch.load(args.Net_save_path + 'Netbase_r4_4_100.pkl')
        net.load_state_dict(net_dict)
        net.eval()
        net.to(device)
        if os.path.exists(args.cutpatch_path):
            num = len(os.listdir(args.cutpatch_path))
            print('test patch has existed', num)
        else:
            num = cutCAVEPieces_Test(args.data_path, args.cutpatch_path, args.test_patch_size, args.ratio)
        num_start = 1
        num_end = num
        for j in range(num_start, num_end + 1):
            path = args.cutpatch_path + str(j) + '.mat'
            data = sio.loadmat(path)
            ms = data['Z']
            lrhs = data['Y']
            lrhs = np.transpose(lrhs, (2, 0, 1))
            lrhs = torch.from_numpy(lrhs).type(torch.FloatTensor).unsqueeze(0).to(device)
            ms = np.transpose(ms, (2, 0, 1))
            ms = torch.from_numpy(ms).type(torch.FloatTensor).unsqueeze(0).to(device)
            out = net(lrhs, ms)
            out1 = out.cpu().squeeze().detach().numpy()
            out1 = np.transpose(out1, (1, 2, 0))
            out1[out1 < 0] = 0.0
            out1[out1 > 1] = 1.0
            checkFile(args.patchtest_path)
            sio.savemat(args.patchtest_path + str(j) + '.mat', {'hs': out1})
            print('%d has finished' % j)
        checkFile(args.results_path)
        merge_Cave_test(args.patchtest_path, args.results_path, args.test_patch_size, args.hs_band)
        quality_mesure_fun(args.results_path, args.data_path)

    @classmethod  # overlap
    def test_piece(self):
        net = model.Net(args.hs_band, args.ms_band)
        net_dict = torch.load(args.Net_save_path + 'Netbase_r4_4_180.pkl')
        net.load_state_dict(net_dict)
        net.eval()
        net.to(device)
        stride = 16
        run_time = 0
        piece_size = 64
        for i in range(args.start, args.end + 1):
            mat = sio.loadmat(args.data_path + '%d.mat' % i)
            print(mat['label'].shape)
            tY = mat['Y']
            tZ = mat['Z']
            output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            start = time.perf_counter()
            for x in range(0, tZ.shape[0] - piece_size + 1, stride):
                for y in range(0, tZ.shape[1] - piece_size + 1, stride):
                    end_x = x + piece_size
                    if end_x + stride > tZ.shape[0]:
                        end_x = tZ.shape[0]
                    end_y = y + piece_size
                    if end_y + stride > tZ.shape[1]:
                        end_y = tZ.shape[1]
                    itY = tY[x // args.ratio:end_x // args.ratio, y // args.ratio:end_y // args.ratio, :]
                    itZ = tZ[x:end_x, y:end_y, :]

                    itY = np.transpose(itY, axes=(2, 0, 1))
                    itY = np.expand_dims(itY, axis=0)
                    itZ = np.transpose(itZ, axes=(2, 0, 1))
                    itZ = np.expand_dims(itZ, axis=0)
                    itY = torch.tensor(itY).to(device)
                    itZ = torch.tensor(itZ).to(device)
                    itY, itZ = itY.to(torch.float), itZ.to(torch.float)
                    tmp = net(itY, itZ)
                    tmp = tmp.cpu().detach().numpy()
                    tmp = np.squeeze(tmp)
                    tmp = np.transpose(tmp, axes=(1, 2, 0))
                    output[x:end_x, y:end_y, :] += tmp
                    num_sum[x:end_x, y:end_y, :] += 1
            output = output / num_sum
            end = time.perf_counter()
            run_time += end - start
            checkFile(args.results_path)
            # sio.savemat(args.results_path + '%d.mat' % i, {'hs': output})
            print('test: %d has finished' % i)
        print('Time: %ss' % (run_time / (args.end - args.start + 1)))
        quality_mesure_fun(args.results_path, args.data_path, args.start, args.end, args.ratio)

    @classmethod
    def efficiency(self):
        net = model.Net(args.hs_band, args.ms_band)
        net_dict = torch.load(args.Net_save_path + 'Netbase_r4_200.pkl')
        net.load_state_dict(net_dict)
        # flops
        input1 = torch.randn(args.batch_size, args.hs_band, 8, 8)
        input2 = torch.randn(args.batch_size, args.ms_band, 32, 32)
        flops, params = profile(net, (input1, input2))
        macs, params = clever_format([flops, params], "%.3f")
        print(macs, params)


if __name__ == '__main__':
    torch.clear_autocast_cache()
    # arguments
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--hs_band", type=int, default=31, help="hs_band")
    parser.add_argument("--ms_band", type=int, default=3, help="ms_band 1 or 3")
    parser.add_argument("--ratio", type=int, default=4, help="ratio")
    parser.add_argument("--lr", type=int, default=0.0001, help="learning_rate")
    parser.add_argument("--test_patch_size", type=int, default=64)
    ##### CAVE #####
    parser.add_argument("--start", type=int, default=21, help="test start")
    parser.add_argument("--end", type=int, default=32, help="test end")
    parser.add_argument("--train_path", type=str, default='/home/lh/Desktop/zzzer/data/cave_64r4/pieces/train/',
                        help="train_path")
    parser.add_argument("--Net_save_path", type=str, default='./Models_CAVE/')
    parser.add_argument("--results_path", type=str, default='./CAVE_HSr4/')
    parser.add_argument("--data_path", type=str, default='/home/lh/Desktop/zzzer/data/cave_64r4/sim/')


    args = parser.parse_args()
    torch.cuda.empty_cache()
    gan_fuse = CD_UNet()
    # gan_fuse.train()
    gan_fuse.test_piece()
    # gan_fuse.efficiency()


