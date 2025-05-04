import sys
sys.path.append(r"../")
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
import os
sys.path.append(os.path.dirname(os.path.abspath("D:\major\AMER_Compressed\AMER\AMER\data_iterator.py")))
from data_iterator import dataIterator
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = [0] if torch.cuda.is_available() else []
dictionaries = ['../dictionary.txt']
hidden_size = 256
batch_size_t = 1
maxlen = 100

def load_dict(dictFile):
    with open(dictFile) as fp:
        lines = fp.readlines()
    lexicon = {}
    for l in lines:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

def for_test(x_t):
    h_mask_t = []
    w_mask_t = []

    encoder = densenet121()
    attn_decoder1 = AttnDecoderRNN(hidden_size, 112, dropout_p=0.5)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
        attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)

    encoder = encoder.to(device)
    attn_decoder1 = attn_decoder1.to(device)

    encoder.load_state_dict(torch.load('../model/encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl', map_location=device))
    attn_decoder1.load_state_dict(torch.load('../model/attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl', map_location=device))

    encoder.eval()
    attn_decoder1.eval()

    x_t = Variable(x_t.to(device))
    x_mask = torch.ones(x_t.size()[0], x_t.size()[1], x_t.size()[2], x_t.size()[3]).to(device)
    x_t = torch.cat((x_t, x_mask), dim=1)
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    h_mask_t.append(int(x_real_high))
    w_mask_t.append(int(x_real_width))
    x_real = x_t[0][0].view(x_real_high, x_real_width)
    output_highfeature_t = encoder(x_t)

    x_mean_t = float(torch.mean(output_highfeature_t))

    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).to(device)
    decoder_hidden_t = decoder_hidden_t * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)

    prediction = torch.zeros(batch_size_t, maxlen)
    decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).to(device)
    attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).to(device)
    decoder_attention_t_cat = []

    for i in range(maxlen):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
            decoder_input_t,
            decoder_hidden_t,
            output_highfeature_t,
            output_area_t,
            attention_sum_t,
            decoder_attention_t,
            dense_input,
            batch_size_t,
            h_mask_t,
            w_mask_t,
            gpu
        )

        decoder_attention_t_cat.append(decoder_attention_t[0].detach().cpu().numpy())
        topv, topi = torch.max(decoder_output, 2)
        if torch.sum(topi) == 0:
            break
        decoder_input_t = topi.view(batch_size_t)

        prediction[:, i] = decoder_input_t

    k = numpy.array(decoder_attention_t_cat)
    x_real = numpy.array(x_real.cpu().data)

    prediction = prediction[0]

    prediction_real = []
    for ir in range(len(prediction)):
        if int(prediction[ir]) == 0:
            break
        prediction_real.append(worddicts_r[int(prediction[ir])])
    prediction_real.append('<eol>')

    prediction_real_show = numpy.array(prediction_real)

    return k, prediction_real_show
