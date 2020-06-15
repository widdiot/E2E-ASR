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
from asr import Dataset
from torch_utils import np2tensor
from torch_utils import pad_list
from train_utils import save_checkpoint
from model_wp import Transducer
from tqdm import tqdm
from edit_distance import compute_wer
from frame_stacking import stack_frame

# from DataLoader import SequentialLoader

parser = argparse.ArgumentParser(description='PyTorch LSTM CTC Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--epoch', type=int,
                    help='evaluate from model saved after this epoch')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--n_stack', type=int, default=4,
                    help='num frames to stack along feat dim')
parser.add_argument('--n_skip', type=int, default=3,
                    help='num frames to skip along time dim')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true',
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/rnnt_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str,
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

trainset = Dataset(corpus='english',
                   tsv_path="/home/new-kaldi/kaldi/egs/eng2k/s5/data/dataset/train_halfk.tsv",
                   dict_path="/home/new-kaldi/kaldi/egs/englaw/s5/data/dict/train_wpbpe15000.txt",
                   unit='wp',
                   wp_model="/home/new-kaldi/kaldi/egs/englaw/s5/data/dict/train_bpe15000.model",
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
                   skip_thought=False)
evalset = Dataset(corpus='hindi',
                 tsv_path="/home/new-kaldi/kaldi/egs/englaw/s5/data/dataset/eval_law.tsv",
                 dict_path="/home/new-kaldi/kaldi/egs/englaw/s5/data/dict/train_wpbpe15000.txt",
                 unit='wp',
                 wp_model="/home/new-kaldi/kaldi/egs/englaw/s5/data/dict/train_bpe15000.model",
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
                 skip_thought=False)



vocab = trainset.vocab
print(vocab)
#batch, is_new_epoch = trainset.next()
#xs = [np2tensor(x).float() for x in batch['xs']]
#xlens = torch.IntTensor([len(x) for x in batch['xs']])
#xs = pad_list(xs, 0.0)
#ys = batch['ys']
#_ys = [np2tensor(np.fromiter(y, dtype=np.int64), -1) for y in ys]  # // TODO vishay optimize for gpu
#ys_out_pad = pad_list(_ys, 0).long()
#ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
# TODO use config file
model = Transducer(81*args.n_stack, vocab, 512, 4, 1024, 2, args.dropout, bidirectional=args.bi)

print(model)
for param in model.parameters():
    torch.nn.init.uniform(param, -0.1, 0.1)
if args.init: model.load_state_dict(torch.load(args.init))
if args.initam: model.encoder.load_state_dict(torch.load(args.initam))
if args.cuda: model.cuda()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=.9)


# data set
# trainset = SequentialLoader('train', args.batch_size)
# devset = SequentialLoader('dev', args.batch_size)


# model = EncoderDecoder(vocab)
def removeDuplicates(S):
    S = list(S)
    n = len(S)
    if (n < 2):
        return
    j = 0  # index of current distinct character
    # Traversing string
    for i in range(n):
        # If current character S[i]
        # is different from S[j]
        if S[j] != S[i]:
            j += 1
            S[j] = S[i]
    # Putting string termination character.
    j += 1
    S = S[:j]
    str1 = ''.join(S)
    return str1


def eval(epoch):
    recog_dir = args.out
    ref_trn_save_path = recog_dir + '/ref_epoch_' + str(epoch) + '.trn'
    hyp_trn_save_path = recog_dir + '/hyp_epoch_' + str(epoch) + '.trn'
    wer, cer = 0, 0
    n_sub_w, n_ins_w, n_del_w = 0, 0, 0
    n_sub_c, n_ins_c, n_del_c = 0, 0, 0
    n_word, n_char = 0, 0
    pbar = tqdm(total=len(evalset))
    f_hyp = open(hyp_trn_save_path, 'w')
    f_ref = open(ref_trn_save_path, 'w')
    losses = []
    is_new_epoch = 0
    #    for xs, ys, xlen, ylen in devset:
    step = 0
    while True:
        batch, is_new_epoch = evalset.next()
#        if is_new_epoch:
#            break
        xs, ys, xlens = batch['xs'], batch['ys'], batch['xlens']
        xs = [stack_frame(x, args.n_stack, args.n_skip) for x in xs]
        xs = [np2tensor(x).float() for x in xs]
        xlen = torch.IntTensor([len(x) for x in xs])
        xs = pad_list(xs, 0.0).cuda()
        _ys = [np2tensor(np.fromiter(y, dtype=np.int64), -1) for y in ys]
        ys_out_pad = pad_list(_ys, 0).long().cuda()
        ylen = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
        # xs = Variable(torch.FloatTens is:open or(xs), volatile=True).cuda()
        # ys = Variable(torch.LongTensor(ys), volatile=True).cuda()
        # xlen = Variable(torch.IntTensor(xlen)); ylen = Variable(torch.IntTensor(ylen))
        model.eval()
        #logging.info('================== Evaluation Mode =================')
        loss = model(xs, ys_out_pad, xlen, ylen)
        loss = float(loss.data) * len(xlen)
        losses.append(loss)
        step += 1  # //TODO vishay un-hardcode the batch size
        best_hyps_id, _ = model.greedy_decode(xs)
       # print(batch['text'],len(batch['xs']))
        for b in range(len(batch['xs'])):
            ref = batch['text'][b]
            hyp = evalset.idx2token[0](best_hyps_id[b])
           # hyp = removeDuplicates(hyp)
            # Write to trn
            utt_id = str(batch['utt_ids'][b])
            speaker = str(batch['speakers'][b]).replace('-', '_')
            if hyp is None:
                hyp = "none"
            f_ref.write(ref + ' (' + speaker + '-' + utt_id + ')\n')
            f_hyp.write(str((hyp + ' (' + speaker + '-' + utt_id + ')\n').encode('utf-8')))
#            logging.info('utt-id: %s' % utt_id)
#            logging.info('Ref: %s' % ref)
#            logging.info('Hyp: %s' % hyp)
#            logging.info('-' * 150)

           # if 'char' in devset.unit:  # //TODO this is only for char unit
                # Compute WER
            wer_b, sub_b, ins_b, del_b = compute_wer(ref=ref.split(' '),
                                                     hyp=hyp.split(' '),
                                                     normalize=False)
            wer += wer_b
            n_sub_w += sub_b
            n_ins_w += ins_b
            n_del_w += del_b
            n_word += len(ref.split(' '))

                # Compute CER
            cer_b, sub_b, ins_b, del_b = compute_wer(ref=list(ref),
                                                     hyp=list(hyp),
                                                     normalize=False)
            cer += cer_b
            n_sub_c += sub_b
            n_ins_c += ins_b
            n_del_c += del_b
            n_char += len(ref)

        pbar.update(len(batch['xs']))
        if is_new_epoch:
            break

    pbar.close()

    # Reset data counters
    evalset.reset()

    
    wer /= n_word
    n_sub_w /= n_word
    n_ins_w /= n_word
    n_del_w /= n_word
    
    

    cer /= n_char
    n_sub_c /= n_char
    n_ins_c /= n_char
    n_del_c /= n_char

    logging.info('WER (%s): %.2f %%' % (evalset.set, wer))
    logging.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_w, n_ins_w, n_del_w))
    logging.info('CER (%s): %.2f %%' % (evalset.set, cer))
    logging.info('SUB: %.2f / INS: %.2f / DEL: %.2f' % (n_sub_c, n_ins_c, n_del_c))

    # print(step, '/12k  dev')
    return sum(losses) / len(evalset), wer, cer


def train():
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
    for epoch in range(1, args.epochs):
        totloss = 0;
        losses = []
        start_time = time.time()
        # for i, (xs, ys, xlen, ylen) in enumerate(trainset):
        step = 0
        is_new_epoch = 0
        while True:
            batch, is_new_epoch = trainset.next()
            if is_new_epoch:
                break
            xs, ys, xlens = batch['xs'], batch['ys'], batch['xlens']
            xs = [np2tensor(x).float() for x in batch['xs']]
            xlen = torch.IntTensor([len(x) for x in batch['xs']])
            xs = pad_list(xs, 0.0).cuda()
            _ys = [np2tensor(np.fromiter(y, dtype=np.int64), -1) for y in ys]
            ys_out_pad = pad_list(_ys, 0).long().cuda()
            ylen = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int32))
            #   xs = Variable(torch.FloatTensor(xs)).cuda()
            if args.cuda: xs = xs.cuda()
            if args.noise: add_noise(xs)
            #   ys = Variable(torch.LongTensor(ys)).cuda()
            #   xlen = Variable(torch.IntTensor(xlen)); ylen = Variable(torch.IntTensor(ylen))
            model.train()
            optimizer.zero_grad()
            loss = model(xs, ys_out_pad, xlen, ylen)
            loss.backward()
            loss = float(loss.data) * len(xlen)
            totloss += loss;
            losses.append(loss)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()

            step += 1  # //TODO vishay un-hardcode the batch size
            # print(step, '/68k')
            if step % args.log_interval == 0 and step > 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] train_loss %.2f' % (epoch, step, loss))
                totloss = 0
        trainset.reset()
        losses = sum(losses) / len(trainset)
        #val_l, wer, cer = eval(epoch)
        val_l = eval(epoch)
        # logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; wer %.2f ; cer %.2f ; lr %.3e' % (
        #     epoch, time.time() - start_time, losses, val_l, wer, cer, lr
        # ))
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e' % (
            epoch, time.time() - start_time, losses, val_l, lr
        ))
        if val_l < prev_loss:
            prev_loss = val_l
            best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, losses, val_l)
            torch.save(model.state_dict(), best_model)
        else:
            torch.save(model.state_dict(),
                       '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, losses, val_l))
            model.load_state_dict(torch.load(best_model))
            if args.cuda: model.cuda()
            if args.schedule:
                lr /= 2
                adjust_learning_rate(optimizer, lr)


if __name__ == '__main__':
    #train()
    loss, wer, cer = eval(args.epoch)
    print('loss: ', loss, ', wer: ', wer, ', cer: ', cer)
    # loss = model(xs, ys_out_pad, xlens, ylens)
# print(loss)

