#This is the baseline model [PAMA](https://doi.org/10.1007/978-3-031-43987-2_69) used in this paper.

import argparse
import os
import random
import time
import warnings

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import os

import pama_models
from loader import *
from utils import *

torch.autograd.set_detect_anomaly(True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MAE pre-training')
parser.add_argument('--root', default= "",metavar='DIR', 
                    help='path to dataset')
parser.add_argument('--train', default="", type=str, metavar='PATH',
                    help='path to train data_root (default: none)')
parser.add_argument('--test', default="", type=str, metavar='PATH',
                    help='path to test data_root (default: none)')
parser.add_argument('--fold', default=1, type=int, help='fold for val')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs for lr_scheduler')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size , this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='WD', help='Set weight decay coefficient (L2 regularization) for optimizer, default: %(default)s', dest='wd')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--seed', default=2024, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='1', type=int,
                    help='GPU id to use.')
parser.add_argument('--weighted-sample', action='store_true',
                    help='Enable weighted sampling for class imbalance handling (adjusts data loading with sample weights) [default: disabled]')
parser.add_argument('--early_stop', action='store_true',
                    help='Enable early stopping when validation loss plateaus to prevent overfitting [default: disabled]')

# additional configs:
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
parser.add_argument('--depth', default=4, type=int,
                    help='the number of PACA module')
parser.add_argument('--max-size', default=8, type=int,
                    help='')
parser.add_argument('--e-dim', default=256, type=int,
                    help='images input size')
parser.add_argument('--mlp-ratio', type=float, default=4, help='MLP ratio')
parser.add_argument('--max-kernel-num', default=16, type=int,
                    help='images input size')
parser.add_argument('--patch-per-kernel', default=144, type=int,
                    help='images input size')
parser.add_argument('--dropout', default=0., type=float,
                    help='')
parser.add_argument('--attn-dropout', default=0., type=float,
                    help='')
parser.add_argument('--num-classes', default=4, type=int,
                    help='number of classes (default: 4)')
parser.add_argument('--list-classes', default=[0,1,2,3], type=list,
                    help='name list of classes')
parser.add_argument('--model', default='pama_vit_base', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--num-sampled-features', default=2048, type=int,
                        help='Number of sampled features.')
parser.add_argument('--num-test-sampled-features', default=2048, type=int,
                        help='Number of sampled features.')
parser.add_argument('--input_size', default=2048, type=int,
                    help='images input size')
parser.add_argument('--in-chans', default=1280, type=int,
                    help='in_chans')
parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0., type=float,
                    help='Masking ratio (percentage of removed patches).')
parser.set_defaults(norm_pix_loss=False)
parser.add_argument('--save-path', default="",
                    help='Path where save the model checkpoint')

def main():
    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path,'fold='+str(args.fold))
    args.checkpoint = os.path.join(args.save_path, "checkpoints")
    args.checkpoint_matrix = os.path.join(args.save_path, "checkpoint-matrix")
    args.checkpoint_roc = os.path.join(args.save_path, "checkpoint_roc")
    args.checkpoint_csv = os.path.join(args.save_path, "checkpoint_csv")

    if args.checkpoint is not None:
        os.makedirs(args.checkpoint, exist_ok=True)
    if args.checkpoint_matrix:
        os.makedirs(args.checkpoint_matrix, exist_ok=True)
    if args.checkpoint_roc:
        os.makedirs(args.checkpoint_roc, exist_ok=True)
    if args.checkpoint_csv:
        os.makedirs(args.checkpoint_csv, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True

    model = pama_models.__dict__[args.model](num_kernel=args.max_kernel_num, img_size=args.max_size, patch_size=1,
                                               in_chans=args.in_chans,
                                               embed_dim=args.e_dim, depth=args.depth, num_heads=8,
                                               mlp_ratio=args.mlp_ratio, num_classes=args.num_classes,
                                               norm_pix_loss=args.norm_pix_loss,
                                               dropout=args.dropout, attn_dropout=args.attn_dropout)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of requires_grad params (M): %.2f' % (n_parameters / 1.e6))
    n_parameters_full = sum(p.numel() for p in model.parameters())
    print('number of the whole params (M): %.2f' % (n_parameters_full / 1.e6))

    # infer learning rate before changing batch size
    init_lr = args.lr 
    weight_de= args.wd
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_de)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    train_dataset = TrainDataset(args.root, args.train,fold=args.fold,
                                        max_size=args.num_sampled_features, set='linprobe',
                                        max_kernel_num=args.max_kernel_num,
                                        patch_per_kernel=args.patch_per_kernel,
                                        args=args)
    valid_dataset = ValDataset(args.root,args.train,fold=args.fold,
                                        max_size=args.num_sampled_features, set="test",
                                        max_kernel_num=args.max_kernel_num,
                                        patch_per_kernel=args.patch_per_kernel,
                                        args=args)
    test_dataset = TestDataset(args.root,args.test,
                                        max_size=args.num_test_sampled_features,
                                        set="test",
                                        max_kernel_num=args.max_kernel_num,
                                        patch_per_kernel=args.patch_per_kernel,
                                        args=args)
    print("train:", len(train_dataset))
    print("val:", len(valid_dataset))
    print("test:", len(test_dataset))
    
    if args.weighted_sample:
        print('activate weighted sampling')
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_dataset.get_weights(), len(train_dataset), replacement=True
            )
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
            
    tra_val_recorder = Record(os.path.join(args.checkpoint_csv , 'train&val_record.csv'))
    test_recorder = TESTRecord(os.path.join(args.checkpoint_csv , 'test_record.csv'))
    
    print(f"Starting fold {args.fold}/5")
    best_auc = 0.0       # Best AUC score achieved
    best_auc_epoch = 0   # Epoch number of best AUC
    if args.early_stop:
        early_stopping = EarlyStopping(patience=15, delta=0.0001)
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_record = train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        val_record = validate(val_loader, model, criterion, epoch, args)
        tra_val_recorder.update([str(args.fold)] + [str(epoch)] + list(train_record) + list(val_record))

        if epoch>=30 and args.early_stop:
            val_auc=float(val_record[4])
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Stopping training in epoch"+str(epoch))
                break            

        current_auc = float(val_record[4])
        if current_auc > best_auc:
            best_auc = current_auc
            best_auc_epoch = epoch
            best_auc_model_path = '{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint, best_auc_epoch)
            if not os.path.isfile(best_auc_model_path):
                save_checkpoint({
                    'fold': args.fold,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint, best_auc_epoch))
    
    # Test using model selected by best AUC score
    checkpoint = torch.load( best_auc_model_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
        
    test_record =test(test_loader, model, criterion, start_epoch, args) 
    test_recorder.update([str('AUC')]+[str(args.fold)] + [str(start_epoch)] + list(test_record))

def train(train_loader, model, criterion, optimizer, epoch,args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top2],
        prefix='Train: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)

    model.train()

    end = time.time()
    for i, (wsidata, labels, slide_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        device = args.gpu
        if args.gpu is not None:
            wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
            wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
            wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
            token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
            kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        # compute output
        logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=args.mask_ratio)

        loss = criterion(logits, labels)
        # measure accuracy and record loss
        losses.update(loss.item(), wsi_feat.size(0))

        acc = accuracy(logits, labels, topk=(1, 2))
        acc1, acc2 = acc[0], acc[1]

        top1.update(acc1[0], wsi_feat.size(0))
        top2.update(acc2[0], wsi_feat.size(0))
        Y_prob = F.softmax(logits, dim=-1)
        cm.update_matrix(Y_prob, labels)
        auc_metric.update(logits, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    micro_auc, macro_auc, weighted_auc = auc_metric.calc_auc_score()

    print('[Train] train-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

    return '{:.3f}'.format(losses.avg),  '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc)

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)
    # switch to evaluate mode
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        end = time.time()
        for i, (wsidata, labels, slide_ids) in enumerate(val_loader):
            device = args.gpu
            if args.gpu is not None:
                wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
                wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
                wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
                token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
                kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            y_true.append(labels)

            # compute output
            logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=args.mask_ratio)
            loss = criterion(logits, labels)
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)
            # measure accuracy and record loss
            losses.update(loss.item(), wsi_feat.size(0))
            acc = accuracy(logits, labels, topk=(1, 2))
            acc1, acc2 = acc[0], acc[1]

            top1.update(acc1[0], wsi_feat.size(0))
            top2.update(acc2[0], wsi_feat.size(0))

            Y_prob = F.softmax(logits, dim=1)
            cm.update_matrix(Y_prob, labels)
            auc_metric.update(logits, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        class_accuracies=per_class_accuracy(cm.confusion_matrix, args.list_classes)
        accuracy_class_0 = class_accuracies[0]
        accuracy_class_1 = class_accuracies[1]
        accuracy_class_2 = class_accuracies[2]
        accuracy_class_3 = class_accuracies[3]
        
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        
        micro_f1, macro_f1, weighted_f1 = calculate_f1_score(y_true, y_pred, args.list_classes)
        micro_auc, macro_auc, weighted_auc= auc_metric.calc_auc_score()
        print('[Eval] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg), '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg),'{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), '{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3)

def test(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)

    # switch to evaluate mode
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        end = time.time()
        for i, (wsidata, labels, slide_ids) in enumerate(val_loader):
            device = args.gpu
            if args.gpu is not None:
                wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
                wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
                wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
                token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
                kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            y_true.append(labels)

            # compute output
            logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=args.mask_ratio)
            loss = criterion(logits, labels)
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)
            # measure accuracy and record loss
            losses.update(loss.item(), wsi_feat.size(0))
            acc = accuracy(logits, labels, topk=(1, 2))
            acc1, acc2 = acc[0], acc[1]

            top1.update(acc1[0], wsi_feat.size(0))
            top2.update(acc2[0], wsi_feat.size(0))

            Y_prob = F.softmax(logits, dim=1)
            cm.update_matrix(Y_prob, labels)
            auc_metric.update(logits, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        class_accuracies=per_class_accuracy(cm.confusion_matrix, args.list_classes)
        accuracy_class_0 = class_accuracies[0]
        accuracy_class_1 = class_accuracies[1]
        accuracy_class_2 = class_accuracies[2]
        accuracy_class_3 = class_accuracies[3]
        
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        micro_f1, macro_f1, weighted_f1 = calculate_f1_score(y_true, y_pred, args.list_classes)

        cm.plot_confusion_matrix(
            normalize=True, save_path='{}/[Test][{}] Confusion Matrix.jpg'.format(args.checkpoint_matrix, epoch))
        micro_auc, macro_auc, weighted_auc= auc_metric.calc_auc_score()
        auc_metric.plot_every_class_roc_curve(
            os.path.join(args.checkpoint_roc, '[Test][{}]_every_class_roc.png'.format(epoch)))
        auc_class = auc_metric.calc_every_class_auc_score()
        auc_class0, auc_class1, auc_class2, auc_class3 = auc_class[0], auc_class[1], auc_class[2], auc_class[3]
        print('[Test] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg), '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg),'{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), '{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3), '{:.3f}'.format(auc_class0), '{:.3f}'.format(auc_class1), '{:.3f}'.format(auc_class2), '{:.3f}'.format(auc_class3)

if __name__ == '__main__':
    main()
