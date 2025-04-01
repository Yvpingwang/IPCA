import argparse
import os
import random
import time
import warnings
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import os

import IPCA_models_mul
import IPCA_models_sin
from loader import *
from utils import *

torch.autograd.set_detect_anomaly(True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--modality1_root', default= "/media/disk1/wangyuping/BRCA-HER2/HE_feature_plip",metavar='DIR', 
                    help='path to dataset')
parser.add_argument('--modality2_root', default= "/media/disk2/wangyuping/BRCA/IHC_plip_feature",metavar='DIR', 
                    help='path to dataset')
parser.add_argument('--fold', default=1, type=int, help='fold for val')
parser.add_argument('--train', default="/media/disk2/wangyuping/BRCA/USTC_BRCA_trainval(fold).csv", type=str, metavar='PATH',
                    help='path to train data_root (default: none)')
parser.add_argument('--test', default="/media/disk2/wangyuping/BRCA/USTC_BRCA_test.csv", type=str, metavar='PATH',
                    help='path to test data_root (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='WD', help='Set weight decay coefficient (L2 regularization) for optimizer, default: %(default)s', dest='wd')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs for lr_scheduler')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=2024, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='1', type=int,
                    help='GPU id to use.')
parser.add_argument('--weighted-sample', action='store_true',
                    help='')
parser.add_argument('--single-modality', action='store_false',
                        help='single modality used for test.')

# additional configs:
parser.add_argument('--early_stop', action='store_true',
                    help='enable early stopping based on validation metrics (default: disabled)')
parser.add_argument('--early-stop-epoch', default=30, type=int, metavar='N',
                    help='minimum training epochs required before early stopping can trigger (default: %(default)s)')
parser.add_argument('--patience-epoch', default=5, type=int, metavar='N',
                    help='number of consecutive epochs without improvement before stopping (default: %(default)s)')
parser.add_argument('--warmup', action='store_true',
                    help='enable learning rate warmup phase (default: disabled)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of epochs for linear learning rate warmup (default: %(default)s)')


# models configs:
parser.add_argument('--depth', default=2, type=int,
                    help='the number of PACA module')
parser.add_argument('--max-size', default=8, type=int,
                    help='')
parser.add_argument('--e-dim', default=256, type=int,
                    help='images input size')
parser.add_argument('--mlp-ratio', type=float, default=4, help='MLP ratio')
parser.add_argument('--max-kernel-num', default=128, type=int,
                    help='images input size')
parser.add_argument('--ihc-kernel-num', default=128, type=int,
                    help='number of classes (default: 3)')
parser.add_argument('--patch-per-kernel', default=18, type=int,
                    help='images input size')
parser.add_argument('--dropout', default=0., type=float,
                    help='')
parser.add_argument('--attn-dropout', default=0., type=float,
                    help='')
parser.add_argument('--ihc-dropout', default=0., type=float,
                    help='')
parser.add_argument('--ihc-attn-dropout', default=0., type=float,
                    help='')
parser.add_argument('--num-classes', default=4, type=int,
                    help='number of classes (default: 3)')
parser.add_argument('--list-classes', default=[0, 1, 2, 3], type=list,
                    help='name list of classes')
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
parser.add_argument('--model', default='IPCA_vit_base', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--num-sampled-features', default=2048, type=int,
                        help='Number of sampled features.')
parser.add_argument('--ihc-num-sampled-features', default=2048, type=int,
                        help='Number of sampled features.')
parser.add_argument('--in-chans', default=512, type=int,
                    help='in_chans')
parser.add_argument('--mask_ratio', default=0., type=float,
                    help='Masking ratio (percentage of removed patches).')
parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)
parser.add_argument('--save-path', default="/media/disk2/wangyuping/brca_exp_results/new_resulets_full399_data/IPCA_plip/IPCA_new_modle/IPCA_only/results(depth=4,with_he_all&ihc_js_loss_all;max-kernel-num=128,patch-per-kernel=18,ihc-kernel-num=18,mask_ratio=0.,dropout=0.1,attn-dropout=0.1;adamw,wd=0.0001;1e-5;e=0.0001,weighted-sample)/only_kl_loss",
                    help='Path where save the model checkpoint')


def main():
    args = parser.parse_args()
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
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

    model = IPCA_models_mul.__dict__[args.model](num_kernel=args.max_kernel_num, img_size=args.max_size, patch_size=1,
                                               in_chans=args.in_chans,
                                               embed_dim=args.e_dim, depth=args.depth, num_heads=8,
                                               mlp_ratio=args.mlp_ratio, num_classes=args.num_classes,
                                               norm_pix_loss=args.norm_pix_loss,num_ik=args.ihc_kernel_num,
                                               dropout=args.dropout, attn_dropout=args.attn_dropout, ihc_dropout=args.ihc_dropout, ihc_attn_dropout=args.ihc_attn_dropout)
    
    model_eval = IPCA_models_sin.__dict__[args.model](num_kernel=args.max_kernel_num, img_size=args.max_size, patch_size=1,
                                               in_chans=args.in_chans,
                                               embed_dim=args.e_dim, depth=args.depth, num_heads=8,
                                               mlp_ratio=args.mlp_ratio, num_classes=args.num_classes,
                                               norm_pix_loss=args.norm_pix_loss)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_eval = model_eval.cuda(args.gpu)

    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of requires_grad params (M)in model: %.2f' % (n_parameters / 1.e6))
    n_parameters_full = sum(p.numel() for p in model.parameters())
    print('number of the whole params (M)in model: %.2f' % (n_parameters_full / 1.e6))

    # infer learning rate before changing batch size
    init_lr = args.lr 
    weight_de= args.wd
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_de)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

     # Data loading code
    train_dataset = MULTrainDataset(
        args.modality1_root,
        args.modality2_root,
        args.train,
        set='linprobe',
        fold=args.fold,
        max_size=args.num_sampled_features,
        ihc_max_size=args.ihc_num_sampled_features,
        max_kernel_num=args.max_kernel_num,
        patch_per_kernel=args.patch_per_kernel,
        args=args)
    valid_dataset = MULValDataset(
        args.modality1_root,
        args.modality2_root,
        args.train,
        set="test",
        fold=args.fold,
        max_size=args.num_sampled_features,
        ihc_max_size=args.ihc_num_sampled_features,
        max_kernel_num=args.max_kernel_num,
        patch_per_kernel=args.patch_per_kernel,
        args=args)
    test_dataset = MULTestDataset(
            args.modality1_root,
            args.modality2_root,
            args.test,
            set="test",
            max_size=args.num_sampled_features,
            ihc_max_size=args.ihc_num_sampled_features,
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

    if args.warmup:
        # Warmup phase: Linear increase for warmup_epochs
        warmup_epochs = args.warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6 / init_lr, # Initial lr=1e-6
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Main scheduler: Cosine decay to 1e-6 for remaining epochs
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(args.epochs - warmup_epochs),
            eta_min=1e-6  # Minimum learning rate preserves fine-tuning capability
        )

        # Combined scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs ]
        )
    else:
        # Fallback: Standard cosine annealing when no warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max = args.epochs,
            eta_min=1e-6  # Minimum learning rate preserves fine-tuning capability
        )

    if args.single_modality:
        recorder_single = Single_tra_val_Record(os.path.join(args.checkpoint_csv , 'train&val_single_record.csv'))
        single_test_recorder = Single_TESTRecord(os.path.join(args.checkpoint_csv , 'test_single_record.csv'))
    else:
        recorder_multi = Multi_tra_val_Record(os.path.join(args.checkpoint_csv , 'train&val_multi_record.csv'))
        multi_test_recorder = Multi_TESTRecord(os.path.join(args.checkpoint_csv , 'test_multi_record.csv'))
    

    print(f"Starting fold {args.fold}/5")
    current_auc=0.0
    best_auc = 0.0  
    best_auc_epoch = 0  
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience_epoch, delta=0.0001)

    for epoch in range(args.start_epoch, args.end_epoch):
        scheduler.step()

        # train for one epoch
        train_record = train(train_loader, model, criterion, optimizer, epoch, args)
        checkpoint_path = '{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint, epoch)
        save_checkpoint({
            'fold': args.fold,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, False, filename=checkpoint_path)

        # evaluate on validation set
        if args.single_modality:
            load_pretrained_weights(model_eval, checkpoint_path)
            val_record_single = validate_single(val_loader, model_eval, criterion, epoch, args)
            recorder_single.update([str(args.fold)] +[str(epoch)] + list(train_record) + list(val_record_single))
        else:
            val_record_multi = validate_multi(val_loader, model, criterion, epoch, args)
            recorder_multi.update([str(args.fold)] +[str(epoch)] + list(train_record) + list(val_record_multi))

        if epoch>=args.early_stop_epoch and args.early_stop:
            if args.single_modality:
                val_auc=float(val_record_single[4])
            else:
                val_auc=float(val_record_multi[7])
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Stopping training in epoch"+str(epoch))
                break

        if args.single_modality:
            current_auc = float(val_record_single[4])
        else:
            current_auc = float(val_record_multi[7])
            
        if  current_auc > best_auc:
            best_auc =  current_auc
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
    checkpoint_auc = torch.load(best_auc_model_path)
    start_epoch = checkpoint_auc['epoch']
    if args.single_modality:
        load_pretrained_weights(model_eval, checkpoint_path)
        test_record = test_single(test_loader, model_eval, criterion, start_epoch, args)
        single_test_recorder.update([str('AUC')]+[str(args.fold)] + [str(start_epoch)] + list(test_record))
    else:
        val_record_multi = validate_multi(val_loader, model, criterion, epoch, args)
        multi_test_record = test_multi(test_loader, model, criterion, start_epoch, args)
        multi_test_recorder.update([str('AUC')]+[str(args.fold)] + [str(start_epoch)] + list(multi_test_record))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    losses_h = AverageMeter('Loss', ':.4f')
    losses_i = AverageMeter('Loss', ':.4f')
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
    for i, (wsidata, ihc_wsidata, labels, slide_ids) in enumerate(train_loader):
        data_time.update(time.time() - end)
        device = args.gpu
        if args.gpu is not None:
            wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
            wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
            wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
            token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
            kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)

            ihc_wsi_feat = ihc_wsidata[0].float().cuda(args.gpu, non_blocking=True)
            ihc_token_mask = ihc_wsidata[1].int().cuda(args.gpu, non_blocking=True)

        labels = labels.cuda(args.gpu, non_blocking=True)
        # compute output
        logits, kernel_tokens, i_logits = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, ihc_wsi_feat, ihc_token_mask, device,mask_ratio = args.mask_ratio)

        loss_h = criterion(logits, labels)
        loss_i = criterion(i_logits, labels)
        loss = loss_h + loss_i 
        losses_h.update(loss_h.item(), wsi_feat.size(0))
        losses_i.update(loss_i.item(), wsi_feat.size(0))
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

    return '{:.3f}'.format(losses.avg),  '{:.3f}'.format(losses_h.avg), '{:.3f}'.format(losses_i.avg), '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc)


def validate_multi(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    losses_h = AverageMeter('Loss', ':.4f')
    losses_i = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    top1_i = AverageMeter('Acc@1_i', ':6.2f')
    top2_i = AverageMeter('Acc@2_i', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)
    auc_metric_i = AUCMetric(args.list_classes)
    # switch to evaluate mode
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        end = time.time()
        for i, (wsidata, ihc_wsidata, labels, slide_ids) in enumerate(val_loader):
            device = args.gpu
            if args.gpu is not None:
                wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
                wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
                wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
                token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
                kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)

                ihc_wsi_feat = ihc_wsidata[0].float().cuda(args.gpu, non_blocking=True)
                ihc_token_mask = ihc_wsidata[1].int().cuda(args.gpu, non_blocking=True)
        
            labels = labels.cuda(args.gpu, non_blocking=True)
            y_true.append(labels)

            # compute output
            logits, kernel_tokens, i_logits = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, ihc_wsi_feat, ihc_token_mask, device,mask_ratio=0)

            loss_h = criterion(logits, labels)
            loss_i = criterion(i_logits, labels)
            loss = loss_h + loss_i
            losses_h.update(loss_h.item(), wsi_feat.size(0))
            losses_i.update(loss_i.item(), wsi_feat.size(0))
            losses.update(loss.item(), wsi_feat.size(0))
            
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)
            # measure accuracy and record loss
            acc = accuracy(logits, labels, topk=(1, 2))
            acc1, acc2 = acc[0], acc[1]

            acc_i = accuracy(i_logits, labels, topk=(1, 2))
            acc1_i, acc2_i = acc_i[0], acc_i[1]

            top1.update(acc1[0], wsi_feat.size(0))
            top2.update(acc2[0], wsi_feat.size(0))
            top1_i.update(acc1_i[0], wsi_feat.size(0))
            top2_i.update(acc2_i[0], wsi_feat.size(0))

            Y_prob = F.softmax(logits, dim=1)
            cm.update_matrix(Y_prob, labels)
            auc_metric.update(logits, labels)
            auc_metric_i.update(i_logits, labels)

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
        micro_auc_i, macro_auc_i, weighted_auc_i= auc_metric_i.calc_auc_score()
        auc_metric.plot_every_class_roc_curve(
            os.path.join(args.checkpoint_roc, '[Eval_muti][{}]_every_class_roc.png'.format(epoch)))
        print('[Eval_multi] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg), '{:.3f}'.format(losses_h.avg), '{:.3f}'.format(losses_i.avg), '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(top1_i.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), '{:.3f}'.format(micro_auc_i), '{:.3f}'.format(macro_auc_i),'{:.3f}'.format(weighted_auc_i), '{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3)

def validate_single(val_loader, model, criterion, epoch, args):
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
        for i, (wsidata,_, labels, _) in enumerate(val_loader):
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
            logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=0)

            loss = criterion(logits, labels)
            losses.update(loss.item(), wsi_feat.size(0))
            
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)
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
        auc_metric.plot_every_class_roc_curve(
            os.path.join(args.checkpoint_roc, '[Eval_muti][{}]_every_class_roc.png'.format(epoch)))
        print('[Eval_single] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg),  '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), '{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3)


def test_multi(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    losses_h = AverageMeter('Loss', ':.4f')
    losses_i = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    top1_i = AverageMeter('Acc@1_i', ':6.2f')
    top2_i = AverageMeter('Acc@2_i', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)
    auc_metric_i = AUCMetric(args.list_classes)
    # switch to evaluate mode
    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        end = time.time()
        for i, (wsidata, ihc_wsidata, labels, slide_ids) in enumerate(val_loader):
            device = args.gpu
            if args.gpu is not None:
                wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
                wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
                wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
                token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
                kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)

                ihc_wsi_feat = ihc_wsidata[0].float().cuda(args.gpu, non_blocking=True)
                ihc_token_mask = ihc_wsidata[1].int().cuda(args.gpu, non_blocking=True)
        
            labels = labels.cuda(args.gpu, non_blocking=True)
            y_true.append(labels)

            # compute output
            logits, kernel_tokens, i_logits = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, ihc_wsi_feat, ihc_token_mask, device,mask_ratio=0)

            loss_h = criterion(logits, labels)
            loss_i = criterion(i_logits, labels)
            loss = loss_h + loss_i
            losses_h.update(loss_h.item(), wsi_feat.size(0))
            losses_i.update(loss_i.item(), wsi_feat.size(0))
            losses.update(loss.item(), wsi_feat.size(0))
            
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)

            acc = accuracy(logits, labels, topk=(1, 2))
            acc1, acc2 = acc[0], acc[1]
            acc_i = accuracy(i_logits, labels, topk=(1, 2))
            acc1_i, acc2_i = acc_i[0], acc_i[1]

            top1.update(acc1[0], wsi_feat.size(0))
            top2.update(acc2[0], wsi_feat.size(0))
            top1_i.update(acc1_i[0], wsi_feat.size(0))
            top2_i.update(acc2_i[0], wsi_feat.size(0))

            Y_prob = F.softmax(logits, dim=1)
            cm.update_matrix(Y_prob, labels)
            auc_metric.update(logits, labels)
            auc_metric_i.update(i_logits, labels)

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
            normalize=True, save_path='{}/[Test_muti][{}] Confusion Matrix.pdf'.format(args.checkpoint_matrix, epoch))
        micro_auc, macro_auc, weighted_auc= auc_metric.calc_auc_score()
        micro_auc_i, macro_auc_i, weighted_auc_i= auc_metric_i.calc_auc_score()
        auc_metric.plot_every_class_roc_curve(
            os.path.join(args.checkpoint_roc, '[Test_muti][{}]_every_class_roc.pdf'.format(epoch)))
        auc_class = auc_metric.calc_every_class_auc_score()
        auc_class0, auc_class1, auc_class2, auc_class3 = auc_class[0], auc_class[1], auc_class[2], auc_class[3]
        print('[Test_multi] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg), '{:.3f}'.format(losses_h.avg), '{:.3f}'.format(losses_i.avg), '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(top1_i.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), '{:.3f}'.format(micro_auc_i), '{:.3f}'.format(macro_auc_i),'{:.3f}'.format(weighted_auc_i), '{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3), '{:.3f}'.format(auc_class0), '{:.3f}'.format(auc_class1), '{:.3f}'.format(auc_class2), '{:.3f}'.format(auc_class3)

def test_single(val_loader, model, criterion, epoch, args):
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
        for i, (wsidata, _, labels, slide_ids) in enumerate(val_loader):
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
            logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=0)

            loss= criterion(logits, labels)
            losses.update(loss.item(), wsi_feat.size(0))
            Y_hat = torch.argmax(logits, dim=1)
            y_pred.append(Y_hat)

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
            normalize=True, save_path='{}/[Test_muti][{}] Confusion Matrix.pdf'.format(args.checkpoint_matrix, epoch))
        micro_auc, macro_auc, weighted_auc= auc_metric.calc_auc_score()
        auc_metric.plot_every_class_roc_curve(
            os.path.join(args.checkpoint_roc, '[Test_muti][{}]_every_class_roc.pdf'.format(epoch)))
        auc_class = auc_metric.calc_every_class_auc_score()
        auc_class0, auc_class1, auc_class2, auc_class3 = auc_class[0], auc_class[1], auc_class[2], auc_class[3]
        print('[Test_single] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t weighted_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, weighted_auc))

        return '{:.3f}'.format(losses.avg),'{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc),'{:.3f}'.format(micro_f1), '{:.3f}'.format(macro_f1),'{:.3f}'.format(weighted_f1),'{:.3f}'.format(accuracy_class_0),'{:.3f}'.format(accuracy_class_1),'{:.3f}'.format(accuracy_class_2),'{:.3f}'.format(accuracy_class_3), '{:.3f}'.format(auc_class0), '{:.3f}'.format(auc_class1), '{:.3f}'.format(auc_class2), '{:.3f}'.format(auc_class3)

if __name__ == '__main__':
    main()
