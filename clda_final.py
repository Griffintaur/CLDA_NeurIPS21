from __future__ import print_function
import argparse
import os
import shutil
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from utils.simclr import *
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase,VGGBase, Predictor_latent, Predictor_deep_latent, confidnet
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler, adjust_learning_rate


from utils.return_dataset_original_final import *

from utils.loss import entropy, adentropy
from utils.loss import PrototypeLoss, CrossEntropyKLD

from pdb import set_trace as breakpoint

from log_utils.utils import ReDirectSTD


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--Temperature', type=float, default=5, metavar='T',
                    help='temperature (default: 5)')
parser.add_argument('--alpha', type=float, default=4, help='value of alpha')
parser.add_argument('--beta', type=float, default=1, help='value of beta')
parser.add_argument('--tau', type=float, default=0.999, help='value of tau')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--mu', type=int, default=4, help='value of mu')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home', 'visda'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

parser.add_argument('--name', type=str, default='', help='Name')

parser.add_argument('--threshold', type=float, default=0.95, help='loss weight')


parser.add_argument('--log_file', type=str, default='./temp.log',
                    help='dir to save checkpoint')

parser.add_argument('--resume', type=str, default='',
                    help='resume from checkpoint')


args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))


log_file_name = './logs/'+'/'+args.log_file
ReDirectSTD(log_file_name, 'stdout', True)
source_loader, labeled_target_loader, target_loader_val, target_loader_test, target_loader_unl, class_list = return_dataset_balance_self(args)


use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)


torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

simclr_loss= SupervisedConLoss(temperature=5,base_temperature=5)
group_simclr = SupervisedConLoss(temperature=5,base_temperature=5)
params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep_latent(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor_latent(num_class=len(class_list), inc=inc,
                   temp=args.Temperature)
weights_init(F1)
lr = args.lr
G = torch.nn.DataParallel(G).cuda()
F1 = torch.nn.DataParallel(F1).cuda()

args.checkpath = args.checkpath +"_"+args.dataset+"_"+str(args.num)+"_"+"_"+str(args.source)+"_"+str(args.target)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(args.checkpath)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def get_group(logits):
    _ , target = torch.max(logits, dim=-1)
    groups ={}
    for x,y in zip(target, logits):
        group = groups.get(x.item(),[])
        group.append(y)
        groups[x.item()]= group
    return groups

def group_simclr_loss(grp_dict_un,group_ema):
    loss = []
    l_fast =[]
    l_slow =[]
    for key in grp_dict_un.keys():
        if key in group_ema:
            l_fast.append(torch.stack(grp_dict_un[key]).mean(dim=0))
            l_slow.append(group_ema[key])
    if len(l_fast) > 0:
        l_fast = torch.stack(l_fast)
        l_slow = torch.stack(l_slow)
        features = torch.cat([l_fast.unsqueeze(1), l_slow.unsqueeze(1)], dim=1)
        loss = group_simclr(features)
        loss = max(torch.tensor(0.000).cuda(),loss)
    else:
        loss= torch.tensor(0.0).cuda()
    return loss
def funcget_group_centroid(logits,labels,centroid):
        #data_t_unl = next(data_iter_t_unl)
    groups={}
    for x,y in zip(labels, logits):
        group = groups.get(x.item(),[])
        group.append(y)
        groups[x.item()]= group
    for key in groups.keys():
            groups[key]=torch.stack(groups[key]).mean(dim=0)
    if centroid !=None:
        for k,v  in centroid.items():
            if groups != None and k in groups:
                centroid[k]=(1-args.tau)*v + (args.tau)*groups[k]
            else:
                centroid[k]=v
    if groups!= None:
        for k,v in groups.items():
            if k not in centroid:
                centroid[k]=v
    return centroid

def train():

    best_acc_test = 0.0

    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    param_lr_confid = []
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps


    data_source_loader = iter(source_loader)
    len_source_labeled = len(source_loader)

    data_target_loader = iter(labeled_target_loader)
    len_target_labeled = len(labeled_target_loader)

    data_iter_t_unl = iter(target_loader_unl)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0
    counter = 0

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            checkpoint = torch.load(args.resume)
            #start_step = checkpoint['step']

            G.load_state_dict(checkpoint['state_dict_G'])  
            F1.load_state_dict(checkpoint['state_dict_discriminator'])           
            #optimizer_g.load_state_dict(checkpoint['optimizer_G'])
            #optimizer_f.load_state_dict(checkpoint['optimizer_D'])
		
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    avg_group_centroid= None
    for step in range(start_step, all_step):
        optimizer_g = adjust_learning_rate(param_lr_g, optimizer_g, step,initial_lr=args.lr,lr_type='cos',epochs=all_step,default_start=start_step)
        optimizer_f = adjust_learning_rate(param_lr_f, optimizer_f, step,initial_lr=args.lr,lr_type='cos',epochs=all_step,default_start=start_step)
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)

        if step % len_source_labeled == 0:
            data_source_labeled = iter(source_loader)
        if step % len_target_labeled == 0:
            data_target_labeled = iter(labeled_target_loader)

        data_t = next(data_target_labeled)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_source_labeled)

        im_data_s = Variable(data_s[0].cuda())
        gt_labels_s = Variable(data_s[1].cuda())
        im_data_t = Variable(data_t[0].cuda())
        gt_labels_t = Variable(data_t[1].cuda())
        im_data_tu = Variable(data_t_unl[0].cuda())
        im_data_tu2 = Variable(data_t_unl[1].cuda())
        zero_grad_all()

        data = torch.cat((im_data_s, im_data_t, im_data_tu, im_data_tu2), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)

        output = G(data)
        out1 = F1(output)
        ns = im_data_s.size(0)
        nt = im_data_t.size(0)
        nl = ns + nt
        nu = im_data_tu.size(0)

        feat_source = torch.softmax(out1[:ns],dim=-1)
        feat_target = torch.softmax(out1[ns:nl],dim=-1)
        feat_target_unlabeled = torch.softmax(out1[nl:nl+nu],dim=-1)
        feature_target_unlabled_hard = torch.softmax(out1[nl+nu:],dim=-1)

        loss_c = criterion(out1[:nl], target)
        feat_target_unlabeled_detach = feat_target_unlabeled.detach()
        features = torch.cat([feat_target_unlabeled_detach.unsqueeze(1), feature_target_unlabled_hard.unsqueeze(1)], dim=1)
        simclr_loss_unlabeled = torch.max(torch.tensor(0.000).cuda(),simclr_loss(features))
        if avg_group_centroid is None:
            grp_loss = torch.tensor(0.00).cuda()
            avg_group_centroid={}
        else:
            grp_unlabeld = get_group(feat_target_unlabeled) 
            grp_loss = group_simclr_loss(grp_unlabeld,avg_group_centroid) 
        loss_comb = loss_c +  args.alpha*simclr_loss_unlabeled + args.beta*grp_loss

        ###############################
        loss_comb.backward()
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        ######################################
        avg_group_centroid= funcget_group_centroid(feat_source.detach(),gt_labels_s,avg_group_centroid)

        log_train = 'Ep: {} lr: {}, loss_all: {:.6f}, loss_c: {:.6f}, simclr_loss: {:.6f}, grp_loss: {:.6f}'.format(step, lr, \
                loss_comb.item(), loss_c.item(), simclr_loss_unlabeled.item(), grp_loss.item())
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            print(log_train)


        if (step % args.save_interval == 0 or step+1== args.steps) and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            is_train_dsne = True

            if acc_test >= best_acc_test:
                best_acc_test = acc_test
                is_best = True
                counter = 0
            else:
                is_best=False
                counter += 1
                
            print('best acc test %f' % (best_acc_test))
            G.train()
            F1.train()


        if step % args.save_interval == 0 and step > 0:
            print('saving model')
            filename = os.path.join(args.checkpath, 
                        "{}_{}_"
                        "to_{}.pth.tar".
                        format(args.log_file, args.source,
                                           args.target))
            state = {'step': step + 1,
                'state_dict_G': G.state_dict(),
                'state_dict_discriminator': F1.state_dict(),
                'optimizer_G' : optimizer_g.state_dict(),
                'optimizer_D': optimizer_f.state_dict()
                }
            save_checkpoint(filename, state,is_best)

def save_checkpoint(filename,state, is_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = Variable(data_t[0].cuda())
            gt_labels_t = Variable(data_t[1].cuda())
            feat = G(im_data_t)
            output1 = F1(feat)
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.4f}%)'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()

