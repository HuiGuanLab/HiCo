import argparse
import os
import random

import warnings
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

import torch.optim

import torch.utils.data

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from hico.hi_encoder import DownstreamEncoder


from dataset import get_finetune_training_set,get_finetune_validation_set


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')
parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')
parser.add_argument('--finetune-skeleton-representation', default='joint', type=str,
                    help='which skeleton-representation to use for downstream training')

parser.add_argument('--knn-neighbours', default=None, type=int,
                    help='number of neighbours used for KNN.')

best_acc1 = 0

# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ",child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('PC weight initial finished!')

def load_moco_encoder_q(model,pretrained):

        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            print("message",msg)

            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))    

def knn(data_train, data_test, label_train, label_test, nn=9):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)
    print("Number of KNN Neighbours = ",nn)
    print("training feature and labels",data_train.shape,len(label_train))
    print("test feature and labels",data_test.shape,len(label_test))

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)

    return acc


def test_extract_hidden(model, data_train, data_eval):
    model.eval()
    for ith, (ith_data,ith_data2, label) in enumerate(data_train):
        print(ith)
        input_tensor = ith_data.cuda()
        input_tensor2 = ith_data2.cuda()
        
        en_hi = model(input_tensor,input_tensor2,knn_eval=True)

        if ith == 0:
            label_train = label
            hidden_array_train = en_hi

        else:
            label_train = torch.cat((label_train, label))
            hidden_array_train = torch.cat((hidden_array_train, en_hi))

    model.eval()
    for ith, (ith_data,ith_data2,  label) in enumerate(data_eval):
        print(ith)
        input_tensor = ith_data.cuda()
        input_tensor2 = ith_data2.cuda()

        en_hi = model(input_tensor,input_tensor2, knn_eval=True)
        en_hi = en_hi
        if ith == 0:
            hidden_array_eval = en_hi
            label_eval = label
        else:
            label_eval = torch.cat((label_eval, label))
            hidden_array_eval = torch.cat((hidden_array_eval, en_hi))

    return hidden_array_train, hidden_array_eval, label_train, label_eval


def clustering_knn_acc(model, train_loader, eval_loader,knn_neighbours=1):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)

    knn_acc_1 = knn(hi_train.cpu(), hi_eval.cpu(), label_train, label_eval, nn=knn_neighbours)

    return knn_acc_1



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1


    # training dataset
    from options import options_retrieval as options 
    if args.finetune_dataset== 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset== 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()

    opts.train_feeder_args['input_representation'] = args.finetune_skeleton_representation
    opts.test_feeder_args['input_representation'] = args.finetune_skeleton_representation

    # create model
    model  = DownstreamEncoder(**opts.encoder_args)
    print(model)
    print("options",opts.encoder_args,opts.train_feeder_args,opts.test_feeder_args)
    if not args.pretrained:
        weights_init(model)


    if args.pretrained:
        # freeze all layers  
        for name, param in model.named_parameters():
                param.requires_grad = False

    # load from pre-trained  model
    load_moco_encoder_q(model,args.pretrained)

    
    model = model.cuda()

    # cudnn.benchmark = True

    # Data loading code
    train_dataset = get_finetune_training_set(opts)
    val_dataset   = get_finetune_validation_set(opts)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,drop_last=False)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=False)

    # Extract frozen features of  the  pre-trained query encoder
    # evaluate a KNN  classifier on extracted features
    acc1 = clustering_knn_acc(model,train_loader,val_loader,knn_neighbours=args.knn_neighbours)

    print("KNN retrieval acc = ",acc1)

if __name__ == '__main__':
    main()

