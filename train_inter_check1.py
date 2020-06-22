import configparser
orgs = configparser.ConfigParser()
orgs.read('rnnt.conf')
import shutil
import os
import time
import random
import argparse
import logging
import numpy as np
import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F
import kaldi_io
from asr1 import Dataset
from torch_utils import np2tensor
from torch_utils import pad_list
from train_utils import save_checkpoint
from model_wp1 import Transducer
from tqdm import tqdm
from edit_distance import compute_wer
from frame_stacking import stack_frame

parser = argparse.ArgumentParser(description='PyTorch LSTM CTC Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N',
                    help='the epoch number from which you want to resume training')
parser.add_argument('--n_stack', type=int, default=4,
                    help='num frames to stack along feat dim')
parser.add_argument('--n_skip', type=int, default=3,
                    help='num frames to skip along time dim')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true',
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--ckpt-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/rnnt_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')
parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'args'), 'w') as f:
    f.write(str(args))
if args.stdout:
    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S',
                        filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)


def total_parameters(model):
    nparamsT = 0
    nparams = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            nparamsT += p.view(-1).size(0)
        nparams += p.view(-1).size(0)
    return nparamsT, nparams


def save_ckp(state, checkpoint_dir, best_model=None):  
    ##best_model only assigned if at end of epoch  this loss better than prev__loss
    epoch = state['epoch']
    step = state['step']
    if best_model:   ## ie end of epoch
        f_path = os.path.join(checkpoint_dir,best_model)
    else:           ## ie middle of epoch ckpt
        f_path = os.path.join( checkpoint_dir,'checkpoint_'+str(epoch)+'_'+str(step))
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    epoch = checkpoint['epoch']
    step = checkpoint['step'] + 1
    offset = checkpoint['offset']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, epoch, step, offset


def eval(model):
    devset = Dataset(corpus='english',
                     tsv_path=orgs['paths']['dev_tsv'],
                     dict_path=orgs['paths']['dict'],
                     unit='wp',
                     wp_model=orgs['paths']['wp_model'],
                     batch_size=args.batch_size,  # * args.n_gpus,
                     n_epochs=args.epochs,
                     min_n_frames=40,
                     max_n_frames=2000,
                     sort_by='input',
                     short2long=True,
                     sort_stop_epoch=100,
                     dynamic_batching=True,
                     subsample_factor=1,
                     discourse_aware=False,
                     skip_thought=False,
                     offset=0,
                     epoch=0)

    pbar = tqdm(total=len(devset))
    losses = []
    is_new_epoch = 0
    step = 0
    while True:
        batch, is_new_epoch = devset.next()
        if is_new_epoch:
            break
        utt = batch['utt_ids']
        xs, ys, xlens = batch['xs'], batch['ys'], batch['xlens']
        xs = [stack_frame(x, args.n_stack, args.n_skip) for x in xs]
        xs = [np2tensor(x).float() for x in xs]
        xlen = torch.IntTensor([len(x) for x in xs])
        xs = pad_list(xs, 0.0).cuda()
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), -1) for y in ys]
        ys_out_pad = pad_list(_ys, 0).long().cuda()
        ylen = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        model.eval()
        loss = model(xs, ys_out_pad, xlen, ylen)
        loss = float(loss.sum().data) * len(xlen)
        losses.append(loss)
        step += 1  # //TODO vishay un-hardcode the batch size


        pbar.update(len(batch['xs']))
    pbar.close()

    # Reset data counters
    devset.reset()

    return sum(losses) / len(devset) #, wer, cer


def train():

### initialize model definition and dataparallel
    with open(orgs['paths']['dict'],encoding='utf-8') as f:
        lines = f.read().splitlines()

    vocab = len(lines) +1
    model_base = Transducer(81*args.n_stack, vocab, 512, 3, 1024, 2, args.dropout, bidirectional=args.bi)
    model = nn.DataParallel(model_base)
    print(model)


### if starting training from start, log the num_parameters and uniform init the values
    if not args.init:
        Trainable,Total = total_parameters(model)
        logging.info("Trainable %.2f M parameters" % (Trainable / 1000000))
        logging.info("Total %.2f M parameters" % (Total / 1000000))
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

    if args.cuda: model.cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=.9)



## if resuming training, load ckpt, load dataset with offset

    if args.init:
        model, optimizer, start_epoch, start_step, start_offset = load_ckp(args.init, model, optimizer)
    else:
        start_epoch = 0
        start_step = 0
        start_offset = 0

    trainset = Dataset(corpus='english',
                       tsv_path=orgs['paths']['train_tsv'],
                       dict_path=orgs['paths']['dict'],
                       unit='wp',
                       wp_model=orgs['paths']['wp_model'],
                       batch_size=args.batch_size,  # * args.n_gpus,
                       n_epochs=args.epochs,
                       min_n_frames=40,
                       max_n_frames=2000,
                       sort_by='input',
                       short2long=True,
                       sort_stop_epoch=100,
                       dynamic_batching=True,
                       subsample_factor=1,
                       discourse_aware=False,
                       skip_thought=False,
                       offset=start_offset,
                       epoch=start_epoch)


    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def add_noise(x):
        dim = x.shape[-1]
        noise = torch.normal(torch.zeros(dim), 0.075)
        if x.is_cuda: noise = noise.cuda()
        x.data += noise

    prev_loss = 2000
    best_model = None

    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        if start_offset > 0 and epoch == start_epoch:
            print('training epoch #'+str(epoch)+' from #'+str(start_offset)+' example ...')
        else:
            print('training epoch #'+str(epoch)+' from start ...')
            start_step = 0
        totloss = 0;
        offset = start_offset
        losses = []
        start_time = time.time()
        step = start_step
        is_new_epoch = 0
        tbar = tqdm(total=len(trainset))
        while True:
            batch, is_new_epoch = trainset.next()
            if is_new_epoch:
                break
            xs, ys, xlens = batch['xs'], batch['ys'], batch['xlens']
            xs = [stack_frame(x, args.n_stack, args.n_skip) for x in xs]
            xs = [np2tensor(x).float() for x in xs]
            xlen = torch.IntTensor([len(x) for x in xs])
            xs = pad_list(xs, 0.0).cuda()
            _ys = [np2tensor(np.fromiter(y, dtype=np.int64), -1) for y in ys]
            ys_out_pad = pad_list(_ys, 0).long().cuda()
            ylen = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
            if args.cuda: xs = xs.cuda()
            if args.noise: add_noise(xs)
            model.train()
            optimizer.zero_grad()
            loss = model( xs, ys_out_pad, xlen, ylen)
            loss.sum().backward()
            loss = float(loss.sum().data) * len(xlen)
            totloss += loss;
            losses.append(loss)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            offset += len(batch['xs'])
            step += 1  # //TODO vishay un-hardcode the batch size
            if step % args.ckpt_interval == 0 and step > 0:
                checkpoint = {'epoch':epoch, 'offset':offset, 'step':step, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                if not os.path.exists(os.path.join(args.out,'ckpt')):
                    os.mkdir(os.path.join(args.out,'ckpt'))
                save_ckp(checkpoint,os.path.join(args.out,'ckpt'))                

            if step % args.log_interval == 0 and step > 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] train_loss %.2f' % (epoch, step, loss))
                totloss = 0
            tbar.update(len(batch['xs']))
        tbar.close()
        trainset.reset()
        losses = sum(losses) / len(trainset)
        print('evaluating epoch #'+str(epoch)+'...')
        val_l = eval(model)
        
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e' % (
            epoch + args.resume_epoch, time.time() - start_time, losses, val_l, lr
        ))
        if val_l < prev_loss:
            prev_loss = val_l
            best_model = 'params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format( epoch + args.resume_epoch, losses, val_l)
            ##when ckpting for end of epoch, send epoch+1 so that the start_epoch when loading ckpt is the next one. and step is 0 as the new start_step
            checkpoint = {'epoch':epoch+1, 'offset':offset, 'step':0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if not os.path.exists(os.path.join(args.out,'models')):
                os.mkdir(os.path.join(args.out,'models'))
            save_ckp(checkpoint,os.path.join(args.out,'models'),best_model=best_model)

#            torch.save(model.state_dict(), best_model)
#            torch.save(model.module.state_dict(),best_model+'_base')     #think this can be loaded for inference into the model_base without wrapping with data parallel.
        else:
            rejected_model =  'params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format( epoch + args.resume_epoch, losses, val_l)
            checkpoint = {'epoch':epoch+1, 'offset':offset, 'step':0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if not os.path.exists(os.path.join(args.out,'models')):
                os.mkdir(os.path.join(args.out,'models'))
            save_ckp(checkpoint,os.path.join(args.out,'models'),best_model=rejected_model)
            print('rejecting this epoch, bcoz',val_l,'>',prev_loss,'loading model::',best_model)
            model, optimizer, _, _,_  = load_ckp(os.path.join(args.out,'models',best_model), model, optimizer)
            if args.cuda: model.cuda()
            if args.schedule:
                lr /= 2
                adjust_learning_rate(optimizer, lr)


#    t0 = time.time()
train()
#    t1 = time.time()
#    print('time to finish one epoch, no parallel',t1-t0)


