import torch
import torch.nn as nn
import os
import json
from PIL import Image
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    dataset_name = config.dataset.train._base_.NAME
    if dataset_name == 'PCN':
        print_log('Experiment on PCN', logger=logger)
        resume_file1 = ''
        if resume_file1 != '':
            if os.path.isfile(resume_file1):
                print("=> loading checkpoint '{}'".format(resume_file1))
                checkpoint = torch.load(resume_file1)
                # start_epoch = checkpoint['epoch'] + 1
                # iteration = checkpoint['iter']
                # print(checkpoint['grnet'])
                base_model.module.grnet_.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['grnet'].items()})
                # model.module.grnet_.load_state_dict(checkpoint['grnet'])
                # autoencoder.load_state_dict(checkpoint['net'])
                # optimizer_AE.load_state_dict(checkpoint['optim'])
                for parma in base_model.module.grnet_.parameters():
                    parma.requires_grad = False
                print('Model structure:')
                for param_tensor in base_model.state_dict():
                    print(param_tensor, '\t', base_model.state_dict()[param_tensor].size())

    elif dataset_name == 'ShapeNet':
        print_log('Experiment on ShapeNet', logger=logger)
        resume_file2 = ''
        if resume_file2 != '':
            if os.path.isfile(resume_file2):
                print("=> loading checkpoint '{}'".format(resume_file2))
                checkpoint = torch.load(resume_file2)
                # start_epoch = checkpoint['epoch'] + 1
                # iteration = checkpoint['iter']
                # print(checkpoint['grnet'])
                base_model.module.grnet_.load_state_dict(
                    {k.replace('module.', ''): v for k, v in checkpoint['base_model'].items()})
                # model.module.grnet_.load_state_dict(checkpoint['grnet'])
                # autoencoder.load_state_dict(checkpoint['net'])
                # optimizer_AE.load_state_dict(checkpoint['optim'])
                for parma in base_model.module.grnet_.parameters():
                    parma.requires_grad = False
                print('Model structure:')
                for param_tensor in base_model.state_dict():
                    print(param_tensor, '\t', base_model.state_dict()[param_tensor].size())

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch)  # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4), int(npoints * 3/4)], fixed_points=None)
                partial = partial.cuda()

            elif dataset_name == 'Completion3d':
                partial = data[0].cuda()
                gt = data[1].cuda()

            elif dataset_name == 'MVP':
                partial = data[0].cuda()
                gt = data[1].cuda()

            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            ret = base_model(partial)

            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)

            _loss = sparse_loss + dense_loss
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger=logger)

        if epoch % args.val_freq == 0 and epoch >= 1:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save checkpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if (config.max_epoch - epoch) < 3:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
    train_writer.close()
    val_writer.close()


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4), int(npoints * 3/4)], fixed_points=None)
                partial = partial.cuda()
            elif dataset_name == 'Completion3d':
                partial = data[0].cuda()
                gt = data[1].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 50 == 0 and epoch % 100 == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input' % idx, input_pc, 200, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')

                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, 200, dataformats='HWC')

            if (idx+1) % 50 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median': 1/2,
    'hard': 3/4
}


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        output_matrix = np.empty((1200, 2048))
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret, f_p, f_v, f_pv = base_model(partial)

                ##### (vis_fea) feature
                # if idx+1 != 8 or 12 or 23 or 24 or 39 or 46 or 60 or 65 or 81 or 114 or 127 or 156 or 163 or 171 or 174 or 181 or 187 or 207 or 210 or 221 or 229 or 248 or 280 \
                #         or 294 or 312 or 358 or 428 or 453 or 458 or 473 or 528 or 545 or 548 or 550 or 614 or 631 or 668 or 671 or 696 or 708 or 729 or 747 or 751 or 761 \
                #         or 780 or 804 or 805 or 812 or 836 or 845 or 851 or 867 or 876 or 886 or 893 or 900 or 903 or 906 or 920 or 926 or 956 or 958 or 967 or 974 or 997 \
                #         or 1009 or 1016 or 1025 or 1038 or 1049 or 1066 or 1092 or 1115 or 1153 or 1186:
                #     output_matrix[idx] = f_pv.cpu().numpy()
                #     output_matrix[idx] = np.zeros(output_matrix.shape[1])
                # else:
                # output_matrix[idx] = f_p.cpu().numpy()
                # output_matrix[idx] = f_v.cpu().numpy()
                output_matrix[idx] = f_pv.cpu().numpy()

                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                     dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # if test_writer is not None and idx % 1 == 0:
                index = idx + 1
                if index % 1000 == 0:
                    vis_path = os.path.join(args.experiment_path, 'visualization')
                    if not os.path.exists(vis_path):
                        os.mkdir(vis_path)

                    input_path = os.path.join(vis_path, 'Input point cloud')
                    if not os.path.exists(input_path):
                        os.mkdir(input_path)
                    input_pc = partial.squeeze().detach().cpu().numpy()
                    input_pc = misc.get_ptcloud_img(input_pc)
                    # test_writer.add_image('Model%02d/Input' % idx, input_pc, epoch_idx, dataformats='HWC')
                    input_pc = Image.fromarray(input_pc)
                    input_pc.save(os.path.join(input_path, 'Model %02d.jpg' % index))

                    sparse_path = os.path.join(vis_path, 'Sparse point cloud')
                    if not os.path.exists(sparse_path):
                        os.mkdir(sparse_path)
                    sparse = coarse_points.squeeze().cpu().numpy()
                    sparse_img = misc.get_ptcloud_img(sparse)
                    # test_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch_idx, dataformats='HWC')
                    sparse_img = Image.fromarray(sparse_img)
                    sparse_img.save(os.path.join(sparse_path, 'Model %02d.jpg' % index))

                    dense_path = os.path.join(vis_path, 'Completed point cloud')
                    if not os.path.exists(dense_path):
                        os.mkdir(dense_path)
                    dense = dense_points.squeeze().cpu().numpy()
                    dense_img = misc.get_ptcloud_img(dense)
                    # test_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch_idx, dataformats='HWC')
                    dense_img = Image.fromarray(dense_img)
                    dense_img.save(os.path.join(dense_path, 'Model %02d.jpg' % index))

                    gt_path = os.path.join(vis_path, 'Real point cloud')
                    if not os.path.exists(gt_path):
                        os.mkdir(gt_path)
                    gt_ptcloud = gt.squeeze().cpu().numpy()
                    gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                    # test_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch_idx, dataformats='HWC')
                    gt_ptcloud_img = Image.fromarray(gt_ptcloud_img)
                    gt_ptcloud_img.save(os.path.join(gt_path, 'Model %02d.jpg' % index))
                    print(index)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]),
                          torch.Tensor([-1, 1, 1]),
                          torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                          torch.Tensor([-1, -1, -1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt)

                    test_losses.update(
                        [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                         dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points, gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)

                    index_1 = idx + 1
                    if index_1 % 1 == 0 and 20000 < index_1 <= 105180:
                        vis_path = os.path.join(args.experiment_path, 'visualization')
                        if not os.path.exists(vis_path):
                            os.mkdir(vis_path)

                        input_path = os.path.join(vis_path, 'Input')
                        if not os.path.exists(input_path):
                            os.mkdir(input_path)
                        input_pc = partial.squeeze().detach().cpu().numpy()
                        input_pc = misc.get_ptcloud_img(input_pc)
                        # test_writer.add_image('Model%02d/Input' % idx, input_pc, epoch_idx, dataformats='HWC')
                        input_pc = Image.fromarray(input_pc)
                        input_pc.save(os.path.join(input_path, 'Model %02d.jpg' % index_1))

                        # sparse_path = os.path.join(vis_path, 'Sparse')
                        # if not os.path.exists(sparse_path):
                        #     os.mkdir(sparse_path)
                        # sparse = coarse_points.squeeze().cpu().numpy()
                        # sparse_img = misc.get_ptcloud_img(sparse)
                        # # test_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch_idx, dataformats='HWC')
                        # sparse_img = Image.fromarray(sparse_img)
                        # sparse_img.save(os.path.join(sparse_path, 'Model %02d.jpg' % index_1))

                        dense_path = os.path.join(vis_path, 'Completed')
                        if not os.path.exists(dense_path):
                            os.mkdir(dense_path)
                        dense = dense_points.squeeze().cpu().numpy()
                        dense_img = misc.get_ptcloud_img(dense)
                        # test_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch_idx, dataformats='HWC')
                        dense_img = Image.fromarray(dense_img)
                        dense_img.save(os.path.join(dense_path, 'Model %02d.jpg' % index_1))

                        gt_path = os.path.join(vis_path, 'Real')
                        if not os.path.exists(gt_path):
                            os.mkdir(gt_path)
                        gt_ptcloud = gt.squeeze().cpu().numpy()
                        gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                        # test_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch_idx, dataformats='HWC')
                        gt_ptcloud_img = Image.fromarray(gt_ptcloud_img)
                        gt_ptcloud_img.save(os.path.join(gt_path, 'Model %02d.jpg' % index_1))
                        print(index_1)

            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx + 1) % 1 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]), logger=logger)

        labels = np.zeros(1200)

        samples_per_category = 150

        for category in range(1, 9):
            start_index = (category - 1) * samples_per_category
            end_index = category * samples_per_category
            labels[start_index:end_index] = category

        tsne = TSNE(n_components=3, random_state=42, learning_rate=100, n_iter=2000)
        embedded_features = tsne.fit_transform(output_matrix)

        cmap = plt.cm.get_cmap('jet')

        # plt.scatter(embedded_features[:, 0], embedded_features[:, 1], s=5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        exclude_samples = [9, 13, 24, 25, 40, 47, 61, 66, 82, 115, 128, 157, 164, 172, 175, 182, 188, 208, 211, 222, 230, 249, 281,
                           295, 313, 359, 429, 454, 459, 474, 529, 546, 549, 551, 615, 632, 669, 672, 697, 709, 730, 748, 752, 762,
                           781, 805, 806, 813, 837, 846, 852, 868, 877, 887, 894, 901, 904, 907, 921, 927, 957, 959, 968, 975, 998,
                           1010, 1017, 1026, 1039, 1050, 1067, 1093, 1116, 1154, 1187]
        mask = np.ones(1200, dtype=bool)
        mask[exclude_samples] = False
        a = embedded_features[mask, 0]
        b = embedded_features[mask, 1]
        c = embedded_features[mask, 2]

        sc = ax.scatter(a, b, c, s=5, c=labels[mask], cmap=cmap)

        cbar = plt.colorbar(sc, pad=0.15, shrink=0.6)

        cbar.set_ticks(np.arange(1, 9))
        cbar.set_ticklabels(['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft'])

        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=8)

        ##### (vis_fea)
        # plt.title("Point feature", fontsize=10)
        # plt.title("Voxel feature", fontsize=10)
        plt.title("Fusion feature", fontsize=10)

        vis_feature_path = os.path.join(args.experiment_path, 'vis_feature_map')
        if not os.path.exists(vis_feature_path):
            os.mkdir(vis_feature_path)

        ##### (vis_fea)
        # plt.savefig(os.path.join(vis_feature_path, 'tsne_visualization_f_p.png'))
        # plt.savefig(os.path.join(vis_feature_path, 'tsne_visualization_f_v.png'))
        plt.savefig(os.path.join(vis_feature_path, 'tsne_visualization_f_pv.png'))

        if dataset_name == 'KITTI':
            return
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return
