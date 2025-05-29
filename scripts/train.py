# --*--coding:utf-8 --*--
import sys
sys.path.append("..")

import torch
from torchvision.utils import save_image
import os, time, json
import time
from options.train_options import TrainOptions
from models import create_model
import numpy as np
from data_set.load_data import add_np_data, get_np_data_as_groupids, load_np_datagroups, DataType, load_Dataset
from scipy import ndimage
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import math


# device configuration
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def normalize_data(img_np, scale):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (scale * cm + 0.0001)
    img_np[img_np < 0] = 0.0
    img_np[img_np > 1.0] = 1.0
    return img_np


def data_augment(image, label, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    label2 = np.zeros(label.shape, dtype='float32')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -0.5, 0) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_var, 1.0 / scale_var)
        M[:, 2] += shift_var
        for c in range(image.shape[1]):
            image2[i, c] = ndimage.interpolation.affine_transform(image[i, c], M[:, :2], M[:, 2], order=1)
            label2[i, c] = ndimage.interpolation.affine_transform(label[i, c], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        if np.random.uniform() >= 0.67:
            image2[i, :] *= intensity_var
            label2[i, :] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.67:
                image2[i, :] = image2[i, :, ::-1, :]
                label2[i, :] = label2[i, :, ::-1, :]
            elif np.random.uniform() <= 0.33:
                image2[i, :] = image2[i, :, :, ::-1]
                label2[i, :] = label2[i, :, :, ::-1]
    return image2, label2


def transform_aug0():
    return transforms.Compose(
        [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
         transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def transform_aug():
    return transforms.Compose(
        [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
         transforms.ToTensor()])


if __name__ == '__main__':
    # data loader
    train_dataset = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine_20211101/val1_reverse/Cardiac_T2C_train_config.json'
    val_dataset = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine_20211101/val1_reverse/Cardiac_T2C_val_config.json'
    np_data_root = '/research/cbim/medical/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine/np_data_test'

    if not os.path.exists(np_data_root):
        os.mkdir(np_data_root)
        add_np_data(project_data_config_files=train_dataset, data_type='train', model_root=np_data_root)
        add_np_data(project_data_config_files=val_dataset, data_type='validation', model_root=np_data_root)

    training_model_path = '/research/cbim/vast/my389/ailab/models/cardiac_T2C/baseline_ours/ours_test2'
    if not os.path.exists(training_model_path):
        os.mkdir(training_model_path)
    print("." * 30)

    opt = TrainOptions().parse()  # get training options
    opt.model = 'scregrefaffinedeform'
    opt.checkpoints_dir = training_model_path
    opt.input_nc = 1  # add a reference image channel

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    batch_size = 1

    # model.save_networks(0)
    # print(12 / 0)

    train_loss_D_A0_dict = []
    train_loss_D_A1_dict = []
    train_loss_G_A_dict = []
    train_loss_cycle_A_dict = []
    train_loss_reg_dict = []

    validation_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.VALIDATION)
    validation_cines, validation_tags = load_np_datagroups(np_data_root, validation_data_group_ids,
                                                           data_type=DataType.VALIDATION)
    val_dataset = load_Dataset(validation_cines, validation_tags)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
    test_n_batches = len(test_set_loader)

    tag_scale = 5
    cine_scale = 8

    pretrained = True
    temp_start_epoch = -1
    retrain_time_dict = []
    use_pretrained = True
    th = 1000
    while use_pretrained:
        if pretrained:
            start_epoch = temp_start_epoch
        else:
            start_epoch = -1
        break_flag = False
        flag = 1
        total_iters = 0  # the total number of training iterations

        for epoch in range(start_epoch+1, opt.n_epochs + opt.n_epochs_decay + 1):
            if total_iters > 1000: th = 9
            if epoch > 0: th = 6
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data_set loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

            model.update_learning_rate()  # update learning rates in the beginning of every epoch.

            print("epochs = ", epoch)
            print("." * 50)

            nb_combined_groups = 1

            training_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.TRAINING)

            if training_data_group_ids == []:
                print('there is no available training nets data.')

            training_data_combined_groupids_list = []
            combined_groupids = []
            for i, group_id in enumerate(training_data_group_ids):
                if i % nb_combined_groups == 0:
                    combined_groupids = [group_id]
                else:
                    combined_groupids.append(group_id)
                if i % nb_combined_groups == nb_combined_groups - 1 or i == len(training_data_group_ids) - 1:
                    training_data_combined_groupids_list.append(combined_groupids)

            epoch_loss_D_A0 = 0
            epoch_loss_D_A1 = 0
            epoch_loss_G_A = 0
            epoch_loss_cycle_A = 0
            epoch_loss_reg = 0

            group_num = 0

            for combined_groupids in training_data_combined_groupids_list:
                group_num += 1
                # print("training group : {:.0f} ".format(group_num))
                train_cines, train_tags = load_np_datagroups(np_data_root, combined_groupids,
                                                             data_type=DataType.TRAINING)
                train_dataset = load_Dataset(train_cines, train_tags)

                training_set_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                                  shuffle=True)
                train_n_batches = len(training_set_loader)
                nt = 23
                frames = 25

                train_n_batches = train_n_batches * nt
                # in each epoch do ...
                epoch_loss_D_A_0 = 0
                epoch_loss_D_A_1 = 0
                epoch_loss_G_A_0 = 0
                epoch_loss_cycle_A_0 = 0
                epoch_loss_reg_0 = 0

                for i, data in enumerate(training_set_loader):
                    cine0, tag0 = data

                    for t in range(frames - nt, frames):
                        tag = tag0[:, t:t + 1, ::]  # only take the last frame
                        cine = cine0[:, t:t + 1, ::]  # only take the last frame
                        # first normalize the data
                        tag1 = normalize_data(tag, tag_scale)
                        cine1 = normalize_data(cine, cine_scale)

                        # second do augmentation with 0.7 probability
                        if np.random.random() > 0.3:
                            tag1, cine1 = data_augment(tag1, cine1, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1,
                                                       flip=True)
                            tag1 = torch.from_numpy(tag1)
                            cine1 = torch.from_numpy(cine1)

                        tag = to_var(tag1)
                        tag_img = tag.cuda()
                        tag_img = tag_img.float()

                        cine = to_var(cine1)
                        cine_img = cine.cuda()
                        cine_img = cine_img.float()

                        shape = tag_img.shape  # batch_size, seq_length, height, width
                        batch_size = shape[0]
                        seq_length = shape[1]
                        height = shape[2]
                        width = shape[3]
                        tag_img = tag_img.contiguous()
                        tag_img = tag_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                        cine_img = cine_img.contiguous()
                        cine_img = cine_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                        if opt.augment: # for the training of content representation network F
                            tag = tag1.repeat(1, 3, 1, 1)
                            cine = cine1.repeat(1, 3, 1, 1)

                            tag = tag.numpy() * 255
                            cine = cine.numpy() * 255
                            tag = tag.squeeze(0)
                            cine = cine.squeeze(0)
                            tag = tag.transpose((1, 2, 0))
                            cine = cine.transpose((1, 2, 0))

                            tag = Image.fromarray(tag.astype('uint8')).convert('RGB')
                            cine = Image.fromarray(cine.astype('uint8')).convert('RGB')
                            A_aug = transform_aug()(tag)
                            B_aug = transform_aug()(cine)

                            A_aug = A_aug[0, ::]
                            B_aug = B_aug[0, ::]

                            A_aug = torch.unsqueeze(A_aug, 0)
                            B_aug = torch.unsqueeze(B_aug, 0)
                            A_aug = torch.unsqueeze(A_aug, 0)
                            B_aug = torch.unsqueeze(B_aug, 0)

                            A_aug = to_var(A_aug)
                            A_aug_img = A_aug.cuda()
                            A_aug_img = A_aug_img.float()

                            B_aug = to_var(B_aug)
                            B_aug_img = B_aug.cuda()
                            B_aug_img = B_aug_img.float()

                            shape = A_aug_img.shape  # batch_size, seq_length, height, width
                            batch_size = shape[0]
                            seq_length = shape[1]
                            height = shape[2]
                            width = shape[3]
                            A_aug_img = A_aug_img.contiguous()
                            A_aug_img = A_aug_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                            B_aug_img = B_aug_img.contiguous()
                            B_aug_img = B_aug_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                            input_data = {'A': tag_img, 'B': cine_img, 'A_aug': A_aug_img, 'B_aug': B_aug_img}
                        else:
                            input_data = {'A': tag_img, 'B': cine_img}

                        if flag == 1:
                            flag += 1
                            model.data_dependent_initialize(input_data)
                            if start_epoch > -1:
                                model.load_networks(start_epoch)

                        model.set_input(input_data)  # unpack data_set from dataset and apply preprocessing

                        # Note for epoch < opt.stage_1_epochs, we do NOT update the registration network.
                        model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network.

                        printnum = 299

                        if total_iters < 100:
                            printnum = 3
                        if total_iters % printnum == 0:  # display images on visdom and save images to a HTML file

                            real_tag, fake_cine, corr_warped_cine, warped_cine_affine, warped_cine_nonrigid, warped_cine_lag, real_cine = model.get_transfering_results()

                            real_tag = real_tag.squeeze(0)
                            real_tag = real_tag.squeeze(0)

                            fake_cine = fake_cine.squeeze(0)
                            fake_cine = fake_cine.squeeze(0)

                            corr_warped_cine = corr_warped_cine.squeeze(0)
                            corr_warped_cine = corr_warped_cine.squeeze(0)

                            warped_cine_affine = warped_cine_affine.squeeze(0)
                            warped_cine_affine = warped_cine_affine.squeeze(0)

                            warped_cine_nonrigid = warped_cine_nonrigid.squeeze(0)
                            warped_cine_nonrigid = warped_cine_nonrigid.squeeze(0)

                            warped_cine_lag = warped_cine_lag.squeeze(0)
                            warped_cine_lag = warped_cine_lag.squeeze(0)

                            real_cine = real_cine.squeeze(0)
                            real_cine = real_cine.squeeze(0)

                            result_imgs = torch.cat((real_tag, fake_cine, corr_warped_cine, warped_cine_affine,
                                                     warped_cine_nonrigid, warped_cine_lag, real_cine,
                                                     10 * torch.abs(warped_cine_affine - real_cine),
                                                     10 * torch.abs(warped_cine_nonrigid - warped_cine_affine)), 1)

                            save_image(result_imgs, os.path.join(training_model_path, 'result_imgs_iter-{}.png'.format(total_iters)))

                        total_iters += opt.batch_size

                        losses = model.get_current_losses()
                        epoch_loss_D_A_0 += losses['D_real']
                        epoch_loss_D_A_1 += losses['D_fake']
                        epoch_loss_G_A_0 += losses['G_GAN']
                        epoch_loss_cycle_A_0 += losses['G_s']
                        epoch_loss_reg_0 += losses['Reg']
                        reg_loss = losses['Reg']
                        #print('reg_loss')
                        #print(reg_loss)
                        if math.isnan(reg_loss) or reg_loss > th:
                            retrain_time_dict.append(time.asctime(time.localtime(time.time())))
                            np.savetxt(os.path.join(training_model_path, 'retrain_time.txt'), retrain_time_dict, fmt='%s')
                            break_flag = True
                            break
                    if break_flag: break
                if break_flag: break
                epoch_loss_D_A_0 = epoch_loss_D_A_0 / train_n_batches
                epoch_loss_D_A_1 = epoch_loss_D_A_1 / train_n_batches
                epoch_loss_G_A_0 = epoch_loss_G_A_0 / train_n_batches
                epoch_loss_cycle_A_0 = epoch_loss_cycle_A_0 / train_n_batches
                epoch_loss_reg_0 = epoch_loss_reg_0 / train_n_batches

                print("training epoch_loss_D0: {:.6f} ".format(epoch_loss_D_A_0))
                print("training epoch_loss_D1: {:.6f} ".format(epoch_loss_D_A_1))
                print("training epoch_loss_G: {:.6f} ".format(epoch_loss_G_A_0))
                print("training epoch_loss_spatial : {:.6f} ".format(epoch_loss_cycle_A_0))
                print("training epoch_loss_registration : {:.6f} ".format(epoch_loss_reg_0))

                print("." * 20)

                epoch_loss_D_A0 += epoch_loss_D_A_0
                epoch_loss_D_A1 += epoch_loss_D_A_1
                epoch_loss_G_A += epoch_loss_G_A_0
                epoch_loss_cycle_A += epoch_loss_cycle_A_0
                epoch_loss_reg += epoch_loss_reg_0

                # break
            if break_flag: break
            epoch_loss_D_A0 = epoch_loss_D_A0 / group_num
            epoch_loss_D_A1 = epoch_loss_D_A1 / group_num
            epoch_loss_G_A = epoch_loss_G_A / group_num
            epoch_loss_cycle_A = epoch_loss_cycle_A / group_num
            epoch_loss_reg = epoch_loss_reg / group_num

            train_loss_D_A0_dict.append(epoch_loss_D_A0)
            train_loss_D_A1_dict.append(epoch_loss_D_A1)
            train_loss_G_A_dict.append(epoch_loss_G_A)
            train_loss_cycle_A_dict.append(epoch_loss_cycle_A)
            train_loss_reg_dict.append(epoch_loss_reg)

            print("training epoch_loss_D_A0 : {:.6f} ".format(epoch_loss_D_A0))
            print("training epoch_loss_D_A1 : {:.6f} ".format(epoch_loss_D_A1))
            print("training epoch_loss_G_A : {:.6f} ".format(epoch_loss_G_A))
            print("training epoch_loss_spatial : {:.6f} ".format(epoch_loss_cycle_A))
            print("training epoch_loss_registration : {:.6f} ".format(epoch_loss_reg))

            np.savetxt(os.path.join(training_model_path, 'train_loss_D_real_dict.txt'), train_loss_D_A0_dict, fmt='%.6f')
            np.savetxt(os.path.join(training_model_path, 'train_loss_D_fake_dict.txt'), train_loss_D_A1_dict, fmt='%.6f')
            np.savetxt(os.path.join(training_model_path, 'train_loss_G_dict.txt'), train_loss_G_A_dict, fmt='%.6f')
            np.savetxt(os.path.join(training_model_path, 'train_loss_spatial_dict.txt'), train_loss_cycle_A_dict, fmt='%.6f')
            np.savetxt(os.path.join(training_model_path, 'train_loss_registration_dict.txt'), train_loss_reg_dict, fmt='%.6f')

            # model.save_networks(epoch)
            # temp_start_epoch = epoch

            if epoch > 160:
                model.save_networks(epoch)
                temp_start_epoch = epoch
            elif epoch % 20 == 0:
                model.save_networks(epoch)
                temp_start_epoch = epoch

            for i, data in enumerate(test_set_loader):
                if i == epoch % test_n_batches:
                    cine0, tag0 = data
                    # plt.imshow(cine[0, 12, ::])
                    # plt.imshow(tag[0, 12, ::])
                    tag = tag0[:, 24:, ::]  # only take the last frame
                    cine = cine0[:, 24:, ::]  # only take the last frame
                    # first normalize the data
                    tag = normalize_data(tag, tag_scale)
                    cine = normalize_data(cine, cine_scale)

                    # tag = tag.to(device)
                    tag = to_var(tag)
                    tag_img = tag.cuda()
                    tag_img = tag_img.float()

                    # cine = cine.to(device)
                    cine = to_var(cine)
                    cine_img = cine.cuda()
                    cine_img = cine_img.float()

                    shape = tag_img.shape  # batch_size, seq_length, height, width
                    batch_size = shape[0]
                    seq_length = shape[1]
                    height = shape[2]
                    width = shape[3]
                    tag_img = tag_img.contiguous()
                    tag_img = tag_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                    cine_img = cine_img.contiguous()
                    cine_img = cine_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                    if opt.augment:
                        input_data = {'A': tag_img, 'B': cine_img, 'A_aug': tag_img, 'B_aug': cine_img}
                    else:
                        input_data = {'A': tag_img, 'B': cine_img}

                    model.set_input(input_data)  # unpack data_set from dataset and apply preprocessing
                    model.forward()

                    real_tag, fake_cine, corr_warped_cine, warped_cine_affine, warped_cine_nonrigid, warped_cine_lag, real_cine = model.get_transfering_results()

                    real_tag = real_tag.squeeze(0)
                    real_tag = real_tag.squeeze(0)

                    fake_cine = fake_cine.squeeze(0)
                    fake_cine = fake_cine.squeeze(0)

                    corr_warped_cine = corr_warped_cine.squeeze(0)
                    corr_warped_cine = corr_warped_cine.squeeze(0)

                    warped_cine_affine = warped_cine_affine.squeeze(0)
                    warped_cine_affine = warped_cine_affine.squeeze(0)

                    warped_cine_nonrigid = warped_cine_nonrigid.squeeze(0)
                    warped_cine_nonrigid = warped_cine_nonrigid.squeeze(0)

                    warped_cine_lag = warped_cine_lag.squeeze(0)
                    warped_cine_lag = warped_cine_lag.squeeze(0)

                    real_cine = real_cine.squeeze(0)
                    real_cine = real_cine.squeeze(0)

                    result_imgs = torch.cat((real_tag, fake_cine, corr_warped_cine, warped_cine_affine,
                                             warped_cine_nonrigid, warped_cine_lag, real_cine, 10*torch.abs(warped_cine_affine-real_cine), 10*torch.abs(warped_cine_nonrigid-warped_cine_affine)), 1)

                    save_image(result_imgs, os.path.join(training_model_path, 'val_result_imgs_epoch-{}.png'.format(epoch)))

        if break_flag: continue
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        break
