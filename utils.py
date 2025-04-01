import math
import shutil
import numpy as np
import torch
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import csv

import yaml


class Record(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(['Fold', 'Epoch', '[Train]Loss', '[Train]acc1', '[Train]acc2', '[Train]micro_auc', '[Train]macro_auc',
                          '[Train]weighted_auc', '[Val]Loss', '[Val]acc1', '[Val]acc2', '[Val]micro_auc',
                          '[Val]macro_auc', '[Val]weighted_auc', '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3'])

    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)
            
class TESTRecord(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(['index', 'Fold', 'Epoch', '[Val]Loss', '[Val]acc1', '[Val]acc2', '[Val]micro_auc',
                          '[Val]macro_auc', '[Val]weighted_auc', '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3', '[Val]auc_class_0', '[Val]auc_class_1', '[Val]auc_class_2', '[Val]auc_class_3'])

    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)

class Multi_tra_val_Record(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(['Fold', 'Epoch', '[Train]Loss', '[Train]HE_Loss', '[Train]IHC_Loss',  '[Train]acc1', '[Train]acc2', '[Train]micro_auc', '[Train]macro_auc',
                          '[Train]weighted_auc', '[Val]Loss',  '[Val]HE_Loss', '[Val]IHC_Loss',
                          '[Val]acc1', '[Val]acc2', '[Val]ihc_acc1', '[Val]micro_auc', '[Val]macro_auc', '[Val]weighted_auc',
                          '[Val]ihc_micro_auc', '[Val]ihc_macro_auc', '[Val]ihc_weighted_auc', '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3'])

    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)

class Multi_TESTRecord(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(["index",'Fold','Epoch', '[Val]Loss',  '[Val]HE_Loss', '[Val]IHC_Loss',
                          '[Val]acc1', '[Val]acc2', '[Val]ihc_acc1', '[Val]micro_auc', '[Val]macro_auc', '[Val]weighted_auc',
                          '[Val]ihc_micro_auc', '[Val]ihc_macro_auc', '[Val]ihc_weighted_auc', '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3',
                          '[Val]auc_class_0', '[Val]auc_class_1', '[Val]auc_class_2', '[Val]auc_class_3'])
    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)

class Single_tra_val_Record(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(['Fold', 'Epoch', '[Train]Loss', '[Train]HE_Loss', '[Train]IHC_Loss',  '[Train]acc1', '[Train]acc2', '[Train]micro_auc', '[Train]macro_auc',
                          '[Train]weighted_auc', '[Val]Loss', 
                          '[Val]acc1', '[Val]acc2', '[Val]micro_auc', '[Val]macro_auc', '[Val]weighted_auc',
                          '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3'])

    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)

class Single_TESTRecord(object):
    def __init__(self, save_path):
        self.save_path = save_path
        with open(self.save_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(["index",'Fold','Epoch', '[Val]Loss', 
                          '[Val]acc1', '[Val]acc2', '[Val]micro_auc', '[Val]macro_auc', '[Val]weighted_auc',
                          '[Val]micro_f1', '[Val]macro_f1', '[Val]weighted_f1',
                          '[Val]accuracy_class_0','[Val]accuracy_class_1','[Val]accuracy_class_2','[Val]accuracy_class_3',
                          '[Val]auc_class_0', '[Val]auc_class_1', '[Val]auc_class_2', '[Val]auc_class_3'])
            
    def update(self, record):
        with open(self.save_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(record)
            
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        Early stopping mechanism: Stops training when validation AUC doesn't improve for `patience` epochs.
        
        :param patience: Number of epochs to wait for AUC improvement before stopping
        :param delta: Minimum required AUC improvement to be considered significant
        :param save_path: Path to save the best model (if provided)
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0  # Counter for epochs without improvement
        self.best_auc = -np.Inf  # Initialize best AUC (fixed)
        self.early_stop = False
        self.best_model = None  # Stores best model state

    def __call__(self, val_auc):
        """
        Performs early stopping check - call after each validation epoch
        
        :param val_auc: Current epoch's validation AUC score
        :param model: PyTorch model to save (optional)
        """
        if val_auc > self.best_auc + self.delta:  # AUC improved
            self.best_auc = val_auc
            self.counter = 0  # Reset counter
        else:  # No AUC improvement
            self.counter += 1
            # print(f"No AUC improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:  # Trigger early stopping
                self.early_stop = True
                print("Early stopping triggered!")

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def per_class_accuracy(cm, classes):
    cm = cm.numpy()
    class_accuracies = {}
    for i, label in enumerate(classes):
        correct_predictions = cm[i, i]
        total_predictions = np.sum(cm[i, :])
        class_accuracies[label] = correct_predictions / total_predictions if total_predictions > 0 else 0

    return class_accuracies

def calculate_f1_score(y_true, y_pred, labels, average='weighted'):
    """
    Calculate overall F1 scores with different averaging methods.
    
    Args:
        y_true: Ground truth (correct) target values (array-like)
        y_pred: Estimated targets as returned by a classifier (array-like)
        labels: List of all possible label values
        average: {'micro', 'macro', 'weighted'}, default='weighted'
                Averaging method for multiclass F1 calculation:
                - 'micro': Calculate metrics globally
                - 'macro': Calculate metrics for each label, find unweighted mean
                - 'weighted': Calculate metrics for each label, find weighted mean
    
    Returns:
        tuple: (f1_micro, f1_macro, f1_weighted) containing:
            - f1_micro: Micro-averaged F1 score
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted-average F1 score (using specified average method)
    """
    f1_micro = f1_score(y_true, y_pred, labels=labels, average='micro')
    f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro')
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average=average)
    return f1_micro, f1_macro, f1_weighted


class ConfusionMatrix(object):
    def __init__(self, classes):
        self.confusion_matrix = torch.zeros(len(classes), len(classes))
        self.classes = classes

    def update_matrix(self, preds, targets):
        # print(preds)
        preds = torch.max(preds, 1)[1].cpu().numpy()
        # preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        # print("====", preds)
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.confusion_matrix[t, p] += 1

    def plot_confusion_matrix(self, normalize=True, save_path='./Confusion Matrix.jpg'):
        cm = self.confusion_matrix.numpy()
        classes = self.classes
        num_classes = len(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        im = plt.matshow(cm, cmap=plt.cm.Blues)  # cm.icefire
        plt.xticks(range(num_classes), classes)
        plt.yticks(range(num_classes), classes)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        tempMax = 0
        for i in range(len(classes)):
            tempSum = 0
            for j in range(num_classes - 1):
                tempS = cm[i, j] * 100
                tempSum += tempS
                color = 'white' if tempS > 50 else 'black'
                if cm[i, j] != 0:
                    plt.text(j, i, format(tempS, '0.2f'), color=color, ha='center')
            tempS = 100 - tempSum
            tempMax = tempS if tempS > tempMax else tempMax
            color = 'white' if tempS > 50 else 'black'
            if float(format(abs(tempS), '0.2f')) != 0:
                plt.text(num_classes - 1, i, format(abs(tempS), '0.2f'), color=color, ha='center')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        cb.set_ticks(np.linspace(0, tempMax / 100., 6))
        cb.set_ticklabels([str("%.2f" % (100 * l)) for l in np.linspace(0, tempMax / 100., 6)])#by wangyuping
        #cb.set_ticklabels(str("%.2f" % (100 * l)) for l in np.linspace(0, tempMax / 100., 6))

        plt.savefig(save_path)
        plt.close()


class AUCMetric(object):
    def __init__(self, classes):
        self.targets = []
        self.preds = []
        self.classes = np.arange(len(classes))
        self.classes_list = classes

    def update(self, preds, targets):
        preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.preds.append(p)
            self.targets.append(t)

    def calc_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        micro_auc = metrics.roc_auc_score(targets, preds, average='micro')
        macro_auc = metrics.roc_auc_score(targets, preds, average='macro')
        weighted_auc = metrics.roc_auc_score(targets, preds, average='weighted')
        return micro_auc, macro_auc, weighted_auc

    def calc_binary_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(list(self.targets)), classes=self.classes)
        auc = metrics.roc_auc_score(targets, preds[:, 1])

        return auc

    def plot_micro_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        print(preds, "------")
        print(targets)
        fpr, tpr, thresholds, = metrics.roc_curve(targets.ravel(), preds.ravel())
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()

    def calc_every_class_auc_score(self):
        """
        Calculate AUC score for each individual class (one-vs-rest approach).
        
        Computes Area Under the ROC Curve (AUC) for each class separately using 
        the one-vs-rest strategy. This is particularly useful for multi-class 
        classification evaluation.

        Returns:
            list: A list of AUC scores where each element corresponds to a class 
                  in self.classes, in the same order.
        """
        y_pred = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)

        # Compute AUC for each class
        auc_per_class = []
        for i in range(y_pred.shape[1]):
            auc = metrics.roc_auc_score(targets[:, i], y_pred[:, i])
            auc_per_class.append(auc)

        return auc_per_class
    
    def plot_every_class_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        fpr = dict()
        tpr = dict()
        auc = dict()
        if len(self.classes) == 5:
            colors = ["aqua", "darkorange", "cornflowerblue", "navy", "deeppink"]
        if len(self.classes) == 9 or 8:
            colors = ["aqua", "darkorange", "cornflowerblue", "navy", "deeppink", "blue", "purple", "green", "gray"]
        for i, color in zip(range(len(self.classes)), colors):
            fpr[i], tpr[i], thresholds, = metrics.roc_curve(targets[:, i], preds[:, i])
            auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                ls="--",
                color=color,
                lw=2,
                alpha=0.7,
                label="ROC of {0} (area={1:0.2f})".format(self.classes_list[i], auc[i]),
            )

        # plot micro_roc_curve
        fpr_micro, tpr_micro, thresholds_micro, = metrics.roc_curve(targets.ravel(), preds.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)

        plt.plot(fpr_micro, tpr_micro, c='r', lw=2, alpha=0.7,
                 label="AUC (area = {:.3f})".format(auc_micro))
        
        # 计算 macro ROC 曲线和 AUC
        #all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.classes))]))
        #mean_tpr = np.zeros_like(all_fpr)

        #for i in range(len(self.classes)):
        #    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        #mean_tpr /= len(self.classes)

        #fpr_macro = all_fpr
        #tpr_macro = mean_tpr
        #auc_macro = metrics.auc(fpr_macro, tpr_macro)
        #plt.plot(fpr_macro, tpr_macro, c='b', lw=2, alpha=0.7,
        #        label="Macro-average ROC (area = {:.3f})".format(auc_macro))

        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(True)
        plt.grid(ls=':')
        #plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()

    def plot_binary_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        fpr, tpr, thresholds, = metrics.roc_curve(targets.ravel(), preds[:, 1].ravel())
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()

        
class Sensitivity_and_Specificity:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        preds = torch.argmax(preds, 1).cpu().numpy().tolist()
        targets = targets.cpu().numpy().tolist()
        self.preds.extend(preds)
        self.targets.extend(targets)

    def calculate(self):
        sensitivity = {}
        specificity = {}
        for c in range(self.num_classes):
            binary_preds = [1 if p == c else 0 for p in self.preds]
            binary_targets = [1 if t == c else 0 for t in self.targets]

            cm = confusion_matrix(binary_targets, binary_preds, labels=[1, 0])
            TP = cm[0, 0]
            FN = cm[0, 1]
            TN = cm[1, 1]
            FP = cm[1, 0]

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity[c] = sens
            specificity[c] = spec

        return sensitivity, specificity


def load_pretrained_weights(model, checkpoint_path):
    """
    Load pretrained weights from checkpoint, only loading parameters with matching keys and shapes.
    
    Args:
        model (torch.nn.Module): Current model instance to load weights into
        checkpoint_path (str): Path to the checkpoint file
        init_unmatched (bool, optional): Whether to initialize unmatched parameters 
                                        using trunc_normal_ initialization. Default: False
        verbose (bool, optional): Whether to print detailed loading logs. Default: True
    
    Returns:
        torch.nn.Module: Model with loaded pretrained weights
    
    Note:
        - Automatically handles keys with 'module.' prefix (common in DDP-trained models)
        - Uses non-strict loading to ignore shape mismatches
        - Preserves original model parameters for unmatched keys
    """
    if not os.path.isfile(checkpoint_path):
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return model

    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k in state_dict:
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = state_dict[k]

    model_state_dict = model.state_dict()
    loaded_params = {}
    unmatched_keys = []

    for k, v in new_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            loaded_params[k] = v
        else:
            unmatched_keys.append(k)

    model_state_dict.update(loaded_params)
    model.load_state_dict(model_state_dict, strict=False)

    if unmatched_keys:
        print("=> Warning: The following keys from checkpoint were not loaded due to mismatch:")
        for k in unmatched_keys:
            print(f"   - {k}")

    return model


#---->read yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)