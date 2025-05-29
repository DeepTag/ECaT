import sys
sys.path.append("..")

import torch
import torch.optim as optim
from torchvision.utils import save_image
import os, time
import math
import cv2
from scipy import ndimage
from models.registration_net_cas_with_generator_detach import Lagrangian_motion_estimate_net, Diffeo_reg_loss, NCC
import numpy as np
from torch.autograd import Variable
from data_set.load_data import add_np_data, get_np_data_as_groupids, load_np_datagroups, DataType, load_Dataset

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


def train_Cardiac_Tagging_ME_net(net, \
                                 np_data_root, \
                                 batch_size, \
                                 n_epochs, \
                                 learning_rate, \
                                 model_path, \
                                 criterionReg, \
                                 criterionRegFlow):
    net.train()
    net.cuda()
    net = net.float()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    # training start time
    training_start_time = time.time()

    train_loss_dict = []

    tag_scale = 5

    pretrained = False
    pretrained_model = '0_-0.5544_model.pth'
    use_pretrained = True
    while use_pretrained:
        if pretrained:
            net.load_state_dict(torch.load(os.path.join(model_path, pretrained_model)))
            start_epoch = int(pretrained_model.split('_')[0]) + 1
        else:
            start_epoch = 0

        print('start_epoch:')
        print(start_epoch)

        break_flag = False
        total_iters = 0  # the total number of training iterations
        for outer_epoch in range(start_epoch, n_epochs):
            # print training log
            print("epochs = ", outer_epoch)
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
            epoch_loss = 0
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

                frames = 3
                nt = frames

                train_n_batches = train_n_batches * nt
                # in each epoch do ...

                epoch_loss_0 = 0

                for i, data in enumerate(training_set_loader):
                    cine0, tag0 = data

                    tag1 = np.zeros([1, frames, tag0.shape[-2], tag0.shape[-1]])

                    for t in range(0, frames):
                        tag = tag0[:, t:t + 1, ::]  # only take the last frame
                        # first normalize the data
                        tag1[:, t, ::] = normalize_data(tag, tag_scale)

                    # second do augmentation with 0.7 probability
                    if np.random.random() > 0.3:
                        tag1, _ = data_augment(tag1, tag1, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=True)

                    tag1 = torch.from_numpy(tag1)

                    tag = to_var(tag1)
                    tag_img = tag.cuda()
                    img = tag_img.float()

                    y1 = img[:, 1:2, ::]
                    y2 = img[:, 2:3, ::]
                    y = torch.cat([y2, y1], dim=1)

                    x1 = img[:, 0:1, ::]
                    x2 = img[:, 1:2, ::]
                    x = torch.cat([x2, x1], dim=1)

                    shape = x.shape  # batch_size, seq_length, height, width
                    height = shape[2]
                    width = shape[3]
                    x = x.contiguous()
                    x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
                    y = y.contiguous()
                    y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                    # set the param gradients as zero
                    optimizer.zero_grad()
                    # forward pass, backward pass and optimization
                    y_src, y_tgt, lag_y_src, inf_flow, neg_inf_flow, lag_flow = net(y, x)  # unpack data_set from dataset and apply preprocessing

                    a = 5
                    b = 1

                    training_loss = criterionReg(y_src, x) + criterionReg(y_tgt, y) + criterionReg(lag_y_src, x) + \
                                    a * criterionRegFlow(inf_flow) + a * criterionRegFlow(neg_inf_flow) + b * criterionRegFlow(lag_flow)
                    training_loss.backward()
                    optimizer.step()

                    printnum = 299

                    if total_iters < 100:
                        printnum = 3
                    if total_iters % printnum == 0:  # display images on visdom and save images to a HTML file

                        x1 = x1.squeeze(0)
                        x1 = x1.squeeze(0)
                        x2 = x2.squeeze(0)
                        x2 = x2.squeeze(0)

                        result_imgs = torch.cat((x1, lag_y_src[1, ::].squeeze(0), x2, lag_y_src[0, ::].squeeze(0)), 1)

                        save_image(result_imgs, os.path.join(training_model_path, 'result_imgs_iter-{}.png'.format(total_iters)))

                    total_iters += batch_size
                    # statistic
                    epoch_loss_0 += training_loss.item()
                    if math.isnan(epoch_loss_0):
                        break_flag = True
                        break

                if break_flag: break
                epoch_loss_0 = epoch_loss_0 / train_n_batches
                print("training ME_epoch_loss_0 : {:.6f} ".format(epoch_loss_0))
                epoch_loss += epoch_loss_0

            if break_flag: break
            epoch_loss = epoch_loss / group_num
            train_loss_dict.append(epoch_loss)
            np.savetxt(os.path.join(model_path, 'train_loss.txt'), train_loss_dict, fmt='%.6f')

            print("training loss: {:.6f} ".format(epoch_loss))

            if outer_epoch > 60:
                torch.save(net.state_dict(),
                           os.path.join(model_path, '{:d}_{:.4f}_model.pth'.format(outer_epoch, epoch_loss)))
                pretrained_model = '{:d}_{:.4f}_model.pth'.format(outer_epoch, epoch_loss)
            elif outer_epoch % 20 == 0:
                torch.save(net.state_dict(),
                           os.path.join(model_path, '{:d}_{:.4f}_model.pth'.format(outer_epoch, epoch_loss)))
                pretrained_model = '{:d}_{:.4f}_model.pth'.format(outer_epoch, epoch_loss)

        if break_flag: continue
        torch.save(net.state_dict(), os.path.join(model_path, 'end_model.pth'))
        print("Training finished! It took {:.2f}s".format(time.time() - training_start_time))
        break


if __name__ == '__main__':
    # data loader
    train_dataset = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine_20211101/val1_reverse/Cardiac_T2C_train_config.json'
    np_data_root = '/research/cbim/medical/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine/np_data'

    if not os.path.exists(np_data_root):
        os.mkdir(np_data_root)
        add_np_data(project_data_config_files=train_dataset, data_type='train', model_root=np_data_root)

    training_model_path = '/research/cbim/vast/my389/ailab/models/cardiac_T2C/baseline_ours/tagging_reg_test'
    if not os.path.exists(training_model_path):
        os.mkdir(training_model_path)
    print("." * 30)

    n_epochs = 300
    learning_rate = 5e-4
    batch_size = 1
    print("......HYPER-PARAMETERS 4 TRAINING......")
    print("batch size = ", batch_size)
    print("learning rate = ", learning_rate)
    print("." * 30)

    # proposed model
    vol_size = (192, 192)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 2]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    criterionReg = NCC()
    criterionRegFlow = Diffeo_reg_loss()
    train_Cardiac_Tagging_ME_net(net=net,
                                 np_data_root=np_data_root,
                                 batch_size=batch_size,
                                 n_epochs=n_epochs,
                                 learning_rate=learning_rate,
                                 model_path=training_model_path,
                                 criterionReg=criterionReg,
                                 criterionRegFlow=criterionRegFlow.gradient_loss
                                 )







