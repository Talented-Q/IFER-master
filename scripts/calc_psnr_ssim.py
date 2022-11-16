import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import cv2
import skimage
import cv2
from PIL import Image
import glob
from skimage.measure import compare_psnr, compare_ssim
from argparse import ArgumentParser
import pdb
from torch.utils import data
from torchvision import transforms


def calc_psnr(im1, im2):
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans


def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans


def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                          range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind
                 in range(len(pred_image_list))]

    return ssim_list


def validation(val_data_loader, device):
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            # pred_image, h_list = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(input_im, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(input_im, gt))

        # --- Save image --- #

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_val(net, val_data_loader, device, exp_name, category, save_tag=False):
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            pass

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, exp_name, category):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        print(image_name_1)
        utils.save_image(pred_image_images[ind], './results/{}/{}/{}'.format(category, exp_name, image_name_1))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 100

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))

class CMP_dataset(data.Dataset):
    def __init__(self, con_path, real_path, transform_con, transform_real):
        self.con_path = con_path
        self.trans_con = transform_con
        self.trans_real = transform_real
        self.real_path = real_path
    def __getitem__(self, index):
        con_path = self.con_path[index]
        real_path = self.real_path[index]
        pil_con = Image.open(con_path)
        pil_con = self.trans_con(pil_con)
        real_img = Image.open(real_path)
        # anno_img = clean_img.convert('RGB')
        pil_real = self.trans_real(real_img)

        return pil_con, pil_real
    def __len__(self):
        return len(self.con_path)

def get_dataset(con_imgs_path,real_imgs_path, batch, trans_train, trans_test):
    con_imgs_path = glob.glob(con_imgs_path)
    real_imgs_path = glob.glob(real_imgs_path)
    dataset = CMP_dataset(con_imgs_path, real_imgs_path, trans_train, trans_test)
    dataloader = data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    return dataloader 

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--inference_dir', default=r'G:\code\our_batch6_ffhq_retest_inversion\inference_results\*.png', type=str, help='Path to inference imgs')
    parser.add_argument('--real_dir', default=r'E:\PythonProjects\data\ffhq\*.png', type=str, help='Path to inference imgs')
    parser.add_argument('--batch', default=70, type=int)
    args = parser.parse_args()

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()])

    val_loader = get_dataset(args.inference_dir, args.real_dir, args.batch, transformer, transformer)
    psnr, ssim = validation(val_loader,'cuda')

    print(psnr)
    print(ssim)