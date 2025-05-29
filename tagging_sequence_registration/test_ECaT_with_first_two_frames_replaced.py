import sys
sys.path.append("..")

import torch
import os, json, shutil, csv
from data_set.load_data import add_np_data, get_np_data_as_groupids, load_np_datagroups, DataType, \
    load_Dataset
import SimpleITK as sitk
import numpy as np
from surface_distance import metrics
import time
from options.train_options import TrainOptions
from models import create_model
from models.registration_net_cas_with_generator_detach import Lagrangian_motion_estimate_net, miccai2018_net_cc_san_grid_warp
from skimage import measure
from torch.autograd import Variable

# device configuration
def to_var( x, volatile=False ):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile)

def PSNR(fake, real):
    x, y = np.where(real != -1)
    mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)
    if mse < 1.0e-10:
        return 100
    else:
        PIXEL_MAX = 1
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def MAE(fake, real):
    x, y = np.where(real != -1)  # coordinate of target points
    # points = len(x)  #num of target points
    mae = np.abs(fake[x, y] - real[x, y]).mean()

    return mae / 2

def normalize_data(img_np, scale):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (scale*cm + 0.0001)
    img_np[img_np < 0] = 0.0
    img_np[img_np >1.0] = 1.0
    return img_np

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    model.load_state_dict(w_dict, strict=True)
    return model

def test_Cardiac_Tagging_ME_net(np_data_root, \
                                val_dataset_files, \
                                model_path, \
                                model_name, \
                                dst_root):
    opt = TrainOptions().parse()  # get test options
    opt.model = 'scregrefaffinedeform'

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = model_path
    opt.epoch = model_name
    opt.isTrain = False

    opt.input_nc = 1  # add a reference image channel


    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()

    # test with eval mode. This only affects layers like batch_norm and dropout.
    model.eval()

    model.load_networks_for_testing(opt.epoch)

    vol_size = (192, 192)
    nf_enc = [16, 32, 32, 32]
    dim = 2
    nf_dec = [32, 32, 32, 32, 16, dim]
    tag_reg_net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    tag_reg_net_model_path = '/research/cbim/vast/my389/ailab/models/cardiac_T2C/baseline_ours/tagging_reg/112_-0.6419_model.pth'

    ME_model = load_dec_weights(tag_reg_net, tag_reg_net_model_path)
    ME_model = ME_model.to(device)
    ME_model.eval()

    with open(val_dataset_files, 'r', encoding='utf-8') as f_json:
        data_config = json.load(f_json)
    if data_config is not None and data_config.__class__ is dict:
        grouped_data_sets = data_config.get('validation')
        if grouped_data_sets.__class__ is not dict: print('invalid validation_config.')

    # check grouped_data_sets
    if grouped_data_sets.__class__ is not dict: print('invalid data config file.')

    group_names = grouped_data_sets.keys()
    val_data_list = []
    for group_name in group_names:
        print('working on %s', group_name)
        filesListDict = grouped_data_sets.get(group_name)
        if filesListDict.__class__ is not dict: continue
        # for sample in tqdm(filesListDict.keys()):
        for sample in filesListDict.keys():
            each_trainingSets = filesListDict.get(sample)
            # list images_data_niix in each dataset
            cine_npz = each_trainingSets.get('cine')
            val_data_list.append(cine_npz)

    validation_data_group_ids = get_np_data_as_groupids(model_root=np_data_root, data_type=DataType.VALIDATION)
    validation_cines, validation_tags  = load_np_datagroups(np_data_root, validation_data_group_ids, data_type=DataType.VALIDATION)
    val_dataset = load_Dataset(validation_cines, validation_tags)
    val_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)

    tag_scale = 5
    cine_scale = 8

    time_list = []


    for i, data in enumerate(test_set_loader):
        cine0, tag0 = data

        cine1 = torch.zeros(cine0.shape)
        tag1 = torch.zeros(tag0.shape)
        for t in range(0, 25):
            tag = tag0[:, t:t + 1, ::]  # only take the last frame
            cine = cine0[:, t:t + 1, ::]  # only take the last frame
            # first normalize the data
            tag1[:, t:t + 1, ::] = normalize_data(tag, tag_scale)
            cine1[:, t:t + 1, ::] = normalize_data(cine, cine_scale)

        # step1: register tag sequence
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
        # print('x.shape')
        # print(x.shape)
        # print('y.shape')
        # print(y.shape)

        testing_start_time = time.time()
        with torch.no_grad():
            y_src, y_tgt, lag_y_src, inf_flow, neg_inf_flow, lag_flow = tag_reg_net(y, x)
        infer_time1 = (time.time() - testing_start_time)

        # step2: tagging translation to cine and registration
        tag1 = tag1.view(-1, 1, height, width)  # [25, 1, 192, 192]
        cine1 = cine1.view(-1, 1, height, width)  # [25, 1, 192, 192]
        tag1[0, ::] = lag_y_src[1, ::].detach()  # replace the first frame image with the warped third frame
        tag1[1, ::] = lag_y_src[0, ::].detach()  # replace the second frame image with the warped third frame

        shape = tag0.shape
        fake_cines = np.zeros([shape[1], shape[-2], shape[-1]])
        warped_cines = np.zeros([shape[1], shape[-2], shape[-1]])
        val_deformation_matrixs = np.zeros([shape[1], 2, shape[-2], shape[-1]])

        for t in range(25):  # the first 2 frames are fake tag grids
            tag = tag1[t:t+1, ::]  # only take the t-th frame
            cine = cine1[t:t+1, ::]  # only take the t-th frame

            # first normalize the data
            tag = normalize_data(tag, tag_scale)
            cine = normalize_data(cine, cine_scale)

            tag = to_var(tag)
            tag_img = tag.cuda()
            tag_img = tag_img.float()

            cine = to_var(cine)
            cine_img = cine.cuda()
            cine_img = cine_img.float()

            shape = tag_img.shape  # batch_size, seq_length, height, width
            height = shape[2]
            width = shape[3]
            tag_img = tag_img.contiguous()
            tag_img = tag_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            cine_img = cine_img.contiguous()
            cine_img = cine_img.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

            input_data = {'A': tag_img, 'B': cine_img}

            model.set_input(input_data)  # unpack data_set from dataset and apply preprocessing
            testing_start_time = time.time()
            model.forward()
            infer_time = (time.time() - testing_start_time)
            time_list.append(infer_time)

            fake_cine, warped_cine, val_deformation_matrix = model.get_eval_results()

            fake_cines[t, ::] = fake_cine.squeeze(0).cpu().detach().numpy()
            warped_cines[t, ::] = warped_cine.squeeze(0).cpu().detach().numpy()
            val_deformation_matrixs[t, ::] = val_deformation_matrix.cpu().detach().numpy()

        val_deformation_matrix_lag0 = val_deformation_matrixs[:,0,::]

        val_deformation_matrix_lag1 = val_deformation_matrixs[:,1,::]


        file_path = val_data_list[i][0]
        root_vec = file_path.split(os.path.sep)
        tgt_root1 = os.path.join(dst_root, root_vec[-4])
        if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
        tgt_root2 = os.path.join(tgt_root1, root_vec[-3])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root3 = os.path.join(tgt_root2, root_vec[-2])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)

        root = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs5/Corrected_Tagging_Cine_masks_combined/'
        val_img_file_root = os.path.join(root, root_vec[-4], root_vec[-3], root_vec[-2])

        val_cine_file = os.path.join(val_img_file_root, 'cine.nii.gz')
        cine_image = sitk.ReadImage(val_cine_file)
        spacing1 = cine_image.GetSpacing()
        origin1 = cine_image.GetOrigin()
        direction1 = cine_image.GetDirection()

        cine_img = sitk.GetImageFromArray(cine0[0,::])
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, 'cine.nii.gz'))

        cine_img = sitk.GetImageFromArray(tag0[0, ::])
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, 'tag.nii.gz'))

        cine_img = sitk.GetImageFromArray(fake_cines)
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, 'fake_cine.nii.gz'))

        cine_img = sitk.GetImageFromArray(warped_cines)
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, 'warped_cine.nii.gz'))

        val_deformation_matrix_lag_img0 = sitk.GetImageFromArray(val_deformation_matrix_lag0)
        val_deformation_matrix_lag_img0.SetSpacing(spacing1)
        val_deformation_matrix_lag_img0.SetOrigin(origin1)
        val_deformation_matrix_lag_img0.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img0, os.path.join(tgt_root3, 'deformation_matrix_x.nii.gz'))

        val_deformation_matrix_lag_img1 = sitk.GetImageFromArray(val_deformation_matrix_lag1)
        val_deformation_matrix_lag_img1.SetSpacing(spacing1)
        val_deformation_matrix_lag_img1.SetOrigin(origin1)
        val_deformation_matrix_lag_img1.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img1, os.path.join(tgt_root3, 'deformation_matrix_y.nii.gz'))

        print('finish: ' + str(i))

    print(np.mean(time_list, axis=0))
    print(np.std(time_list, axis=0, ddof=1))
    return time_list


def test_Cardiac_cine_ME_net_mask(net, mask_root, flow_root, dst_root):
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    for subroot, dirs, files in os.walk(flow_root):
        if len(files) < 3: continue
        root_vec = subroot.split(os.path.sep)
        tgt_root2 = os.path.join(dst_root, root_vec[-3])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root20 = os.path.join(tgt_root2, root_vec[-2])
        if not os.path.exists(tgt_root20): os.mkdir(tgt_root20)
        tgt_root3 = os.path.join(tgt_root20, root_vec[-1])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)


        for file in files:
            if file.endswith('.nii.gz') and 'deformation_matrix_x' in file:
                deformation_matrix_x_img_file = file
            if file.endswith('.nii.gz') and 'deformation_matrix_y' in file:
                deformation_matrix_y_img_file = file
                # break

        my_source_file = os.path.join(subroot, 'cine.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'cine.nii.gz')
        shutil.copy(my_source_file, my_target_file)
        my_source_file = os.path.join(subroot, 'warped_cine.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'warped_cine.nii.gz')
        shutil.copy(my_source_file, my_target_file)
        my_source_file = os.path.join(subroot, 'fake_cine.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'fake_cine.nii.gz')
        shutil.copy(my_source_file, my_target_file)
        my_source_file = os.path.join(subroot, 'tag.nii.gz')
        my_target_file = os.path.join(tgt_root3, 'tag.nii.gz')
        shutil.copy(my_source_file, my_target_file)

        deformation_matrix_x_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_x_img_file))
        deformation_matrix_y_img = sitk.ReadImage(os.path.join(subroot, deformation_matrix_y_img_file))
        deformation_matrix_x = sitk.GetArrayFromImage(deformation_matrix_x_img)
        deformation_matrix_y = sitk.GetArrayFromImage(deformation_matrix_y_img)

        val_mask_file_root = os.path.join(mask_root, root_vec[-3], root_vec[-2], root_vec[-1])

        ED_mask_file = os.path.join(val_mask_file_root, 'c2.nii.gz')
        sitk_ED_mask_image = sitk.ReadImage(ED_mask_file)
        # sitk.WriteImage(sitk_ED_mask_image, os.path.join(tgt_root3, 'c2.nii.gz'))

        ES_mask_file = os.path.join(val_mask_file_root, 't2.nii.gz')
        gt_ES_mask_image = sitk.ReadImage(ES_mask_file)
        # sitk.WriteImage(gt_ES_mask_image, os.path.join(tgt_root3, 't2.nii.gz'))
        gt_ED_mask_image = sitk.GetArrayFromImage(sitk_ED_mask_image)
        gt_ES_mask_image = sitk.GetArrayFromImage(gt_ES_mask_image)

        t, y, x = gt_ES_mask_image.shape
        extra_t = 25 - t
        if extra_t > 0:
            B = np.concatenate((gt_ES_mask_image, np.tile(gt_ES_mask_image[-1, ::], (extra_t, 1, 1))), axis=0)
            gt_ES_mask_image = B

        # wrap input data in a Variable object
        flow_x = torch.from_numpy(deformation_matrix_x).to(device)
        flow_y = torch.from_numpy(deformation_matrix_y).to(device)
        flow_x = flow_x.float()
        flow_y = flow_y.float()

        shape = flow_x.shape  #seq_length, height, width
        height = shape[1]
        width = shape[2]
        x = flow_x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width


        y = flow_y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        # wrap input data in a Variable object
        tag_image = torch.from_numpy(gt_ED_mask_image.astype(float)).to(device)
        # wrap input data in a Variable object
        tag_image = tag_image.float()
        z = tag_image
        z = z.contiguous()
        z0 = z.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        eular_flow = torch.cat((x, y), dim=1) # DeepTag-python

        grids_eular = torch.zeros_like(z0)
        grids_eular = grids_eular.to(device)

        for lm in range(1, 2):
            ones = lm*torch.ones_like(z0)
            zeros = torch.zeros_like(z0)
            z_lm = torch.where(z0 == lm, ones, z0)
            z_lm = torch.where(z_lm != lm, zeros, z_lm)
            if torch.sum(z_lm) > 0.1:
                grid = net(z_lm, eular_flow)
                grid = torch.where(grid > 0, ones, grid)
                grid = torch.where(grid <= 0, zeros, grid)
                grids_eular += grid

        grids = grids_eular[:,0,::]
        resgistered_grids = grids.cpu().detach().numpy()

        spacing1 = deformation_matrix_x_img.GetSpacing()
        origin1 = deformation_matrix_x_img.GetOrigin()
        direction1 = deformation_matrix_x_img.GetDirection()

        resgistered_grids_img = sitk.GetImageFromArray(resgistered_grids)
        resgistered_grids_img.SetSpacing(spacing1)
        resgistered_grids_img.SetOrigin(origin1)
        resgistered_grids_img.SetDirection(direction1)
        sitk.WriteImage(resgistered_grids_img, os.path.join(tgt_root3, 'warped_c2.nii.gz'))

        resgistered_grids_img = sitk.GetImageFromArray(gt_ED_mask_image)
        resgistered_grids_img.SetSpacing(spacing1)
        resgistered_grids_img.SetOrigin(origin1)
        resgistered_grids_img.SetDirection(direction1)
        sitk.WriteImage(resgistered_grids_img, os.path.join(tgt_root3, 'c2.nii.gz'))

        resgistered_grids_img = sitk.GetImageFromArray(gt_ES_mask_image)
        resgistered_grids_img.SetSpacing(spacing1)
        resgistered_grids_img.SetOrigin(origin1)
        resgistered_grids_img.SetDirection(direction1)
        sitk.WriteImage(resgistered_grids_img, os.path.join(tgt_root3, 't2.nii.gz'))


        print('finish: ' + str(0))





if __name__ == '__main__':
    # data loader
    val_dataset = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine_20211101/val1_reverse/Cardiac_T2C_new_val_config.json'
    val_np_data_root = '/research/cbim/medical/my389/ailab/data/NYU_Cine_Tagging_pairs2/Tag2Cine/new_val_np_data'

    if not os.path.exists(val_np_data_root):
        os.mkdir(val_np_data_root)
        add_np_data(project_data_config_files=val_dataset, data_type='validation', model_root=val_np_data_root)

    test_model_path = '/research/cbim/vast/my389/ailab/models/cardiac_T2C/baseline_ours/ours_test/'

    test_model = 'test_results_all_models'
    dst_root_all_models = os.path.join(test_model_path, test_model)
    if not os.path.exists(dst_root_all_models): os.mkdir(dst_root_all_models)
    print(dst_root_all_models)
    test_model_path222 = os.path.join(test_model_path, 'test_stats_results_all_models')
    if not os.path.exists(test_model_path222): os.mkdir(test_model_path222)

    dice_dic = {}
    test_n = 0

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vol_size = (192, 192)

    for subroot, dirs, files in os.walk(os.path.join(test_model_path, 'Tagging_transfer_to_Untagged_Cine')):
        if len(files) < 4: continue
        for file in files:
            if file.endswith('.pth') and 'net_Reg' in file:
                model_name = file
                epoch_vector = model_name.split('_')
                epoch = epoch_vector[0]
                if int(epoch) < 0 or int(epoch) > 1000: continue
                print('model_name:')
                print(model_name)
                test_n += 1

                test_time = test_Cardiac_Tagging_ME_net(np_data_root=val_np_data_root,
                                         val_dataset_files=val_dataset,
                                         model_path= test_model_path,
                                         model_name = int(epoch),
                                         dst_root=dst_root_all_models)


                net2 = miccai2018_net_cc_san_grid_warp(vol_size)
                net2 = net2.to(device)


                mask_root = '/research/cbim/vast/my389/ailab/data/NYU_Cine_Tagging_pairs5/Corrected_Tagging_Cine_masks_combined/'
                flow_root = dst_root_all_models
                dst_root_masks = os.path.join(test_model_path, test_model + '_masks/')
                if not os.path.exists(dst_root_masks): os.mkdir(dst_root_masks)

                test_Cardiac_cine_ME_net_mask(net=net2, mask_root=mask_root, flow_root=flow_root, dst_root=dst_root_masks)

                # evaluation
                src_root = dst_root_masks
                
                test_model_path111 = '/research/cbim/vast/my389/ailab/models/cardiac_T2C_2/baseline2/tagging_transfer_to_cine_reg_affine_nonrigid_deform_only_reg3_0.774/test_stats_results_all_models_240_0.774/test_results_all_models_240_masks'

                dice_dic[epoch] = []
                dice_dic[epoch].append(int(epoch))
                avg_total_dice = 0
                avg_total_HD = 0
                avg_total_dice_vec = []
                avg_total_HD_vec = []

                for lm in range(1, 2):
                    evaluate_results = {}
                    avg_dice = 0
                    avg_HD = 0
                    dice_vec = []
                    HD_vec = []
                    MAE_s = []
                    PSNR_s = []
                    SSIM_s = []
                    patients = 0
                    for subroot, dirs, files in os.walk(os.path.join(src_root)):
                        if len(files) < 6: continue
                        patient_num_vec = subroot.split('/')
                        p1 = patient_num_vec[-3] + '_' + patient_num_vec[-2]
                        s1 = patient_num_vec[-1]

                        root_vec = subroot.split(os.path.sep)

                        fake_file = os.path.join(subroot, 'fake_cine.nii.gz')
                        real_file = os.path.join(subroot, 'warped_cine.nii.gz')

                        fake_sitk = sitk.ReadImage(fake_file)
                        real_sitk = sitk.ReadImage(real_file)

                        fake_img = sitk.GetArrayFromImage(fake_sitk)
                        real_img = sitk.GetArrayFromImage(real_sitk)

                        s = fake_img.shape

                        for k in range(s[0]):
                            fake = fake_img[k,::]
                            real = real_img[k,::]
                            mae = MAE(fake, real)
                            psnr = PSNR(fake, real)
                            ssim = measure.compare_ssim(fake, real)
                            MAE_s.append(mae)
                            PSNR_s.append(psnr)
                            SSIM_s.append(ssim)

                        gt_mask_file = os.path.join(subroot, 't2.nii.gz')
                        gt_mask_image = sitk.ReadImage(gt_mask_file)
                        gt_np_mask = np.uint8(sitk.GetArrayFromImage(gt_mask_image))


                        pre_mask_file = os.path.join(subroot, 'warped_c2.nii.gz')
                        pre_mask_image = sitk.ReadImage(pre_mask_file)
                        pre_np_mask = np.uint8(sitk.GetArrayFromImage(pre_mask_image))

                        ones = lm * np.ones(gt_np_mask.shape)
                        zeros = np.zeros(gt_np_mask.shape)
                        mask_gt_lm = np.where(gt_np_mask == lm, ones, gt_np_mask)
                        mask_gt_lm = np.where(mask_gt_lm != lm, zeros, mask_gt_lm)
                        if np.sum(mask_gt_lm) < 0.5: continue  # skip the slice without the cardiac anatomy structure
                        pre_np_mask_lm = np.where(pre_np_mask == lm, ones, pre_np_mask)
                        pre_np_mask_lm = np.where(pre_np_mask_lm != lm, zeros, pre_np_mask_lm)
                        if np.sum(pre_np_mask_lm) < 0.5: continue  # skip the slice without the cardiac anatomy structure
                        mask_gt_lm = np.uint8(mask_gt_lm)
                        pre_np_mask_lm = np.uint8(pre_np_mask_lm)
                        spacing = pre_mask_image.GetSpacing()

                        for i in range(25):
                            mask_gt_lm_1 = mask_gt_lm[i:i+1, ::]

                            pre_np_mask_lm_1 = pre_np_mask_lm[i:i+1, ::]
                            dice = metrics.compute_dice_coefficient(mask_gt=mask_gt_lm_1, mask_pred=pre_np_mask_lm_1)

                            surface_distances = metrics.compute_surface_distances(
                                # mask_gt=np.expand_dims(mask_gt_lm_1, axis=0),
                                mask_gt=mask_gt_lm_1,
                                mask_pred=pre_np_mask_lm_1, spacing_mm=(spacing[2], spacing[1], spacing[0]))
                            H_distance = metrics.compute_robust_hausdorff(surface_distances=surface_distances, percent=95)

                            evaluate_results[p1 + '_' + s1 + '_' + 't'+str(i)] = []
                            evaluate_results[p1 + '_' + s1 + '_' + 't'+str(i)].append(p1 + '_' + s1+ '_' + 't'+str(i))
                            evaluate_results[p1 + '_' + s1 + '_' + 't'+str(i)].append(round(dice, 3))
                            evaluate_results[p1 + '_' + s1 + '_' + 't'+str(i)].append(round(H_distance, 3))
                            dice_vec.append(dice)
                            HD_vec.append(H_distance)

                            patients += 1
                            avg_dice += dice
                            avg_HD += H_distance

                    evaluate_results['avg'] = []
                    evaluate_results['avg'].append('Average')
                    evaluate_results['avg'].append(round(avg_dice / patients, 3))
                    evaluate_results['avg'].append(round(avg_HD / patients, 3))
                    evaluate_results['avg2'] = []
                    evaluate_results['avg2'].append('Average2')
                    evaluate_results['avg2'].append(np.mean(dice_vec))
                    evaluate_results['avg2'].append(np.mean(HD_vec))
                    evaluate_results['std2'] = []
                    evaluate_results['std2'].append('Std2')
                    evaluate_results['std2'].append(np.std(dice_vec, ddof=1))
                    evaluate_results['std2'].append(np.std(HD_vec, ddof=1))

                    dst_out_path = os.path.join(test_model_path222,
                                                'test_results_all_' + epoch + '_' + str(lm) + '_' + str(
                                                    round(avg_dice / patients, 3)))
                    if not os.path.exists(dst_out_path): os.mkdir(dst_out_path)

                    with open(os.path.join(dst_out_path, 'cmr_seg_info_' + str(lm) + '.csv'), 'w',
                              newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for key in evaluate_results.keys():
                            writer.writerow(evaluate_results.get(key))

                    avg_total_dice_vec.append(np.mean(dice_vec))
                    avg_total_HD_vec.append(np.mean(HD_vec))
                    dice_dic[epoch].append(np.mean(MAE_s))
                    dice_dic[epoch].append(np.std(MAE_s, ddof=1))
                    dice_dic[epoch].append(np.mean(PSNR_s))
                    dice_dic[epoch].append(np.std(PSNR_s, ddof=1))
                    dice_dic[epoch].append(np.mean(SSIM_s))
                    dice_dic[epoch].append(np.std(SSIM_s, ddof=1))
                    dice_dic[epoch].append(np.mean(dice_vec))
                    dice_dic[epoch].append(np.std(dice_vec, ddof=1))
                    dice_dic[epoch].append(np.mean(HD_vec))
                    dice_dic[epoch].append(np.std(HD_vec, ddof=1))

                dice_dic[epoch].append(np.mean(test_time, axis=0))
                dice_dic[epoch].append(np.std(test_time, axis=0, ddof=1))

    csv_target_root = os.path.join(test_model_path222, 'Dice_all_models_results.csv')
    with open(csv_target_root, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in dice_dic.keys():
            writer.writerow(dice_dic.get(key))






