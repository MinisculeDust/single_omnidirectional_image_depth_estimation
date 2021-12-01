import time
import argparse
import datetime
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import transforms
from loss import ssim
from load3D60 import getTrainingTestingData
from utils import AverageMeter, DepthNorm
import time
import numpy as np
import evaluate_domain_adaptation
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='End-to-end 360 Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--bs_e', default=16, type=int, help='evaluation batch size')
    parser.add_argument('--maxd', default=10, type=float, help='maximum depth')
    parser.add_argument('--mind', default=0, type=float, help='minimum depth')
    parser.add_argument('--scheduler', default=False, type=bool, help='switch of the scheduler_lr')
    parser.add_argument('--singledomain', default=True, type=bool, help='switch to single domain mode')
    parser.add_argument('--tm', default='', type=str, help='trained-model saving path')
    parser.add_argument('--pkl', default='', type=str, help='pkl results saving path')
    parser.add_argument('--train1', default='', type=str, help='source domain training dataset')
    parser.add_argument('--maxbatch', default=9999, type=int, help='Max Batches')
    parser.add_argument('--comments', default='comments', type=str, help='saved model name (by which dataset or other comments)')
    parser.add_argument('--tdcsv', default='', type=str, help='target domain testing dataset')
    args = parser.parse_args()

    print("the learning rate is: ", args.lr)
    print("the batch size rate is: ", args.bs)
    print("the train1 dataset is: ", args.train1)
    if args.singledomain == False:
        print("the train2 dataset is: ", args.train2)
    else:
        print("No Source Domain2")
        from model_domain_adaptation_single_ResNet50 import Model
    print("the test dataset is: ", args.tdcsv)


    # checking cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The Device is: ', str(device))

    # Create model
    model = Model().to(device)
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
                                               threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=1e-08, eps=1e-08)
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_source_loader1 = getTrainingTestingData(batch_size=batch_size, csv_file=args.train1)
    if args.singledomain == False:
        train_source_loader2 = getTrainingTestingData(batch_size=batch_size, csv_file=args.train2)
    train_target_loader = getTrainingTestingData(batch_size=batch_size, csv_file=args.tdcsv)

    # Loss
    l1_criterion = nn.L1Loss()
    # NLLLOSS: The value corresponding to the output Label is taken out, then is set as negative(to remove the negative sign), and then the mean value is calculated
    # NLLLOSS expects the labels to be in range [0, C-1]
    NLL_criterion = torch.nn.NLLLoss()

    # Train the same number of batches from both datasets
    if args.singledomain == False:
        max_batches = min(len(train_source_loader1), len(train_source_loader2), len(train_target_loader))

    if args.singledomain == True:
        max_batches = min(len(train_source_loader1), len(train_target_loader))

    if args.maxbatch is not None and int(args.maxbatch) < max_batches:
        max_batches = int(args.maxbatch)

    # Check any error during training (especially for loss == Nan)
    with torch.autograd.set_detect_anomaly(True):
        result_list = []
        for epoch in range(args.epochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            N = max_batches

            # Switch to train mode
            model.train()

            end = time.time()

            # iteratively read the dataloader
            train_source_iter = iter(train_source_loader1)
            if args.singledomain == False:
                train_source_iter2 = iter(train_source_loader2)
            train_target_iter = iter(train_target_loader)

            for i in range(max_batches):

                optimizer.zero_grad()

                # load batches one by one
                source_sample_batch = next(train_source_iter)
                if args.singledomain == False:
                    source_sample_batch2 = next(train_source_iter2)
                target_sample_batch = next(train_target_iter)

                ###############################################################
                ###                 Load Source Domain Data                 ###
                ###############################################################
                # Prepare source
                if str(device) == 'cpu':
                    image_source = torch.autograd.Variable(source_sample_batch['image'].to(device)) # ndarray: torch.Size([4, 256, 512, 3])
                    depth_source = torch.autograd.Variable(source_sample_batch['depth'].to(device)) # ndarray: torch.Size([4, 256, 512])
                else:
                    image_source = torch.autograd.Variable(source_sample_batch['image'].cuda())
                    depth_source = torch.autograd.Variable(source_sample_batch['depth'].cuda(non_blocking=True))
                if depth_source.shape[1] != 1:
                    depth_source = depth_source.permute(0, 3, 1, 2) # torch.Size([bs, 1, 128, 256])

                # Normalize source depth
                depth_n_source = DepthNorm(depth_source, minDepth=args.mind, maxDepth=args.maxd)

                ###############################################################
                ###                 Load Source2 Domain Data                ###
                ###############################################################
                if args.singledomain == False:
                    # Prepare source
                    if str(device) == 'cpu':
                        image_source2 = torch.autograd.Variable(
                            source_sample_batch2['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
                        depth_source2 = torch.autograd.Variable(
                            source_sample_batch2['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
                    else:
                        image_source2 = torch.autograd.Variable(source_sample_batch2['image'].cuda())
                        depth_source2 = torch.autograd.Variable(source_sample_batch2['depth'].cuda(non_blocking=True))
                    if depth_source2.shape[1] != 1:
                        depth_source2 = depth_source2.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

                    # Normalize source depth
                    depth_n_source2 = DepthNorm(depth_source2, minDepth=args.mind, maxDepth=args.maxd)


                ###############################################################
                ###                 Load Target Domain Data                 ###
                ###############################################################
                # Prepare target
                if str(device) == 'cpu':
                    image_target = torch.autograd.Variable(
                        target_sample_batch['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
                else:
                    image_target = torch.autograd.Variable(target_sample_batch['image'].cuda())


                # Training progress and GRL lambda
                p = float(i + epoch * N) / (args.epochs * N)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1

                print('Start training...')

                ###############################################################
                ###                 Train on Source Domain1                 ###
                ###############################################################
                y_s_domain = torch.ones(len(image_source), dtype=torch.long).to(device)
                # Predict
                output_s, pre_s_domain = model(image_source.float(), grl_lambda)
                # Compute first source domain loss
                loss_s_domain = NLL_criterion(pre_s_domain, y_s_domain)
                # Compute the end-to-end loss
                l_s_depth = l1_criterion(output_s, depth_n_source)

                # Compute SSIM loss
                l_s_ssim = torch.clamp((1 - ssim(output_s, depth_n_source, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)


                ###############################################################
                ###                  Train on Source Domain2                ###
                ###############################################################
                if args.singledomain == False:
                    y_s2_domain = (torch.ones(len(image_source2), dtype=torch.long) * 2).to(device)
                    # y_s2_domain = torch.ones(len(image_source2), dtype=torch.long).to(device)
                    # Predict
                    output_s2, pre_s2_domain = model(image_source2.float(), grl_lambda)
                    # Compute first source domain loss
                    loss_s2_domain = NLL_criterion(pre_s2_domain, y_s2_domain)
                    # Compute the end-to-end loss
                    l_s2_depth = l1_criterion(output_s2, depth_n_source2)

                    # Compute SSIM loss
                    l_s2_ssim = torch.clamp((1 - ssim(output_s2, depth_n_source2, val_range=1000.0 / 10.0)) * 0.5, 0, 1)


                ###############################################################
                ###                  Train on Target Domain                 ###
                ###############################################################
                y_t_domain = torch.zeros(len(image_target), dtype=torch.long).to(device)
                # Predict
                _, pre_t_domain = model(image_target.float(), grl_lambda)
                # Compute first source domain loss
                loss_t_domain = NLL_criterion(pre_t_domain, y_t_domain)



                # loss includes source ssim, source depth, source domain, target domain
                if args.singledomain == False:
                    loss = (1.0 * l_s_ssim) + (0.1 * l_s_depth) + (1.0 * loss_s_domain) \
                        + (1.0 * l_s2_ssim) + (0.1 * l_s2_depth) + (1.0 * loss_s2_domain) \
                           + (1.0 * loss_t_domain)
                else:
                    loss = (1.0 * l_s_ssim) + (0.1 * l_s_depth) + (1.0 * loss_s_domain) \
                           + (1.0 * loss_t_domain)

                # update recorded losses
                losses.update(loss.data.item(), image_source.size(0))

                # update parameters
                loss.backward()
                optimizer.step()

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

                # Log progress
                if i % 5 == 0:
                    # Print to console
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                    'ETA {eta}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                if (i+1) >= max_batches:
                    print('break')
                    break

            # test the model every 5 epochs
            if (epoch+1) % 5 == 0:
                e = evaluate_domain_adaptation.evaluate(model, args.tdcsv, minDepth=args.mind, maxDepth=args.maxd,
                                                            crop=None,
                                                            batch_size=args.bs_e)
                print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3',
                                                                              'rel', 'rms', 'log_10'))
                print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2],
                                                                                          e[3], e[4], e[5]))
                result_list.append(e)

            if args.scheduler is True:
                scheduler_lr.step(loss.data.item())

    # save the trained-model
    save_path = args.tm + \
                'Ubuntu_' + \
                time.strftime("%Y_%m_%d-%H_%M_UK") \
                + '_epochs-' + str(args.epochs) \
                + '_lr-' + str(args.lr) \
                + '_bs-' + str(args.bs) \
                + '_maxDepth-' + str(args.comments) \
                + '.pth'
    torch.save(model.state_dict(), save_path)
    print('saved the model to:', save_path)
    joblib_save_path = args.pkl + \
                       'Ubuntu_' + \
                       time.strftime("%Y_%m_%d-%H_%M_UK") \
                       + '_epochs-' + str(args.epochs) \
                       + '_lr-' + str(args.lr) \
                       + '_bs-' + str(args.bs) \
                       + '_maxDepth-' + str(args.comments) \
                       + '.pkl'
    joblib.dump(result_list, joblib_save_path)
    print('saved the model to:', joblib_save_path)


if __name__ == '__main__':
    main()
