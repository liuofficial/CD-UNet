import numpy as np
import scipy.io as sio
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def PSNR_GPU(labels, output):
    # print(labels.shape,output.shape)
    s = labels.shape[0]
    # print(s)
    labels = labels.cpu().squeeze().detach().numpy()
    output = output.cpu().squeeze().detach().numpy()
    if s == 1:
        labels = np.transpose(labels, (1, 2, 0))
        output = np.transpose(output, (1, 2, 0))
    else:
        labels = np.transpose(labels, (0, 2, 3, 1))
        output = np.transpose(output, (0, 2, 3, 1))

    return peak_signal_noise_ratio(labels, output)


def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def cutCAVEPieces_Test(mpath, msave_path, path_size, ratio):
    piece_size = path_size
    stride = piece_size
    rows, cols = 512, 512
    num_start = 21
    num_end = 32
    ratio = ratio
    mat_path = mpath
    count = 0
    piece_save_test = msave_path
    checkFile(piece_save_test)
    # print('11111111111111111111')
    for i in range(num_start, num_end + 1):
        mat = sio.loadmat(mat_path + '%d.mat' % i)
        X = mat['label']
        Z = mat['Z']
        Y = mat['Y']
        for x in range(0, rows - piece_size + stride, stride):
            for y in range(0, cols - piece_size + stride, stride):
                data2_piece = Z[x:x + piece_size, y:y + piece_size, :]
                label_piece = X[x:x + piece_size, y:y + piece_size, :]
                data1_piece = Y[x // ratio:(x + piece_size) // ratio, y // ratio:(y + piece_size) // ratio, :]
                count += 1
                sio.savemat(piece_save_test + '%d.mat' % count,
                            {'Y': data1_piece, 'Z': data2_piece, 'X': label_piece})
                print('piece num %d has saved' % count)
        print('%d has finished' % i)
    # print(count)
    print('done')
    return count


def merge_Cave_test(tpath, spath, patch_size, hs_band):
    piece_size = patch_size
    stride = piece_size
    rows, cols = 512, 512
    mat_path = tpath
    count = 0
    save_path = spath
    checkFile(save_path)
    for i in range(0, 12):
        HS = np.zeros((rows, cols, hs_band))
        j = ((rows // patch_size) ** 2) * i + 1
        for x in range(0, rows - piece_size + stride, stride):
            for y in range(0, cols - piece_size + stride, stride):
                mat = sio.loadmat(mat_path + '%d.mat' % j)
                z = mat['hs']
                HS[x:x + piece_size, y:y + piece_size, :] = z
                j = j + 1
        sio.savemat(save_path + str(21 + i) + '.mat', {'hs': HS})
    print(count)
    print('done')


def quality_accessment(out: dict, reference, target, ratio):
    out['cc'] = CC(reference, target)
    out['sam'] = SAM(reference, target)[0]
    out['rmse'] = RMSE(reference, target)
    # out['mrae'] = MRAE(reference, target)
    out['egras'] = ERGAS(reference, target, ratio)
    out['psnr'] = PSNR(reference, target)
    out['ssim'] = SSIM(reference, target)
    return out


def CC(reference, target):
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')
        target_temp = target[:, :, i].flatten(order='F')
        cc = np.corrcoef(ref_temp, target_temp)
        out[i] = cc[0, 1]
    return np.mean(out)


def dot(m1, m2):
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def SAM(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)
    prod_scal = dot(reference, target)
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps
    map = np.arccos(prod_scal / prod_map)
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:0]
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    angolo = np.sum(np.arccos(prod_scal / prod_norm)) / prod_scal.shape[0]
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam, map


# def SSIM_BAND(reference, target):
#     return compare_ssim(reference,target,data_range=1.0)


def SSIM(reference, target):
    # rows,cols,bands = reference.shape
    # mssim = 0
    # for i in range(bands):
    #     mssim += SSIM_BAND(reference[:,:,i],target[:,:,i])
    # mssim /= bands
    # return mssim
    return structural_similarity(reference, target, multichannel=True)


def PSNR(reference, target):
    # max_pixel = 1.0
    # return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))
    return peak_signal_noise_ratio(reference, target)


def RMSE(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def MRAE(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sum(np.divide(np.abs(reference - target), reference, where=reference != 0)) / pixels
    return out


def ERGAS(references, target, ratio):
    rows, cols, bands = references.shape
    d = 1 / ratio
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels
    uk = np.mean(tar_temp, axis=0)
    relative_rmse2 = rmse2 / uk ** 2
    total_relative_rmse = np.sum(relative_rmse2)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)
    return out


def quality_mesure_fun(target_path, reference_path, start, end, ratio):
    out = {}
    average_out = {'cc': 0, 'sam': 0, 'psnr': 0, 'rmse': 0, 'egras': 0, 'ssim': 0}
    # a = np.load('/home/lh/Desktop/zzzer/data/train_list.npy')
    # a = a.tolist()
    # for i in range(1, 16):
    #     if i not in a:
    for i in range(start, end + 1):
        mat = sio.loadmat(reference_path + '%d.mat' % i)
        # print(mat.keys())
        reference = mat['label']
        target = sio.loadmat(target_path + '%d.mat' % i)
        # print(target.keys())
        target = target['hs']
        target = np.float32(target)
        # print(reference.shape, target.shape)
        # target[target < 0] = 0.0
        # target[target > 1] = 1.0
        quality_accessment(out, reference, target, ratio)
        print(out)
        for key in out.keys():
            average_out[key] += out[key]
        print('image %d has finished' % i)
    for key in average_out.keys():
        average_out[key] /= (end - start + 1)
    print(average_out)

