# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-docstring
import keys
train_infofile = 'data/ocr_module/infofile_train_10w.txt'
train_infofile_fullimg = ''
val_infofile = '.data/ocr_module/infofile_test.txt'
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
workers = 4
batchSize = 50
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet)+1
nh = 256
niter = 100
lr = 0.0003
beta1 = 0.5
cuda = True
ngpu = 1
pretrained_model = ''
saved_model_dir = 'src/crnn_models'
saved_model_prefix = 'CRNN-'
use_log = False
remove_blank = False
experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = False
adadelta = False
keep_ratio = False
random_sample = True
dataset='./data/classification_module/classification_data.csv'
Invoice_test=r'./data/e2e_pipeline/Invoice_test.txt'
temp_csv_file='./data/e2e_pipeline/TrainNew.csv'
cat_discount_map='./data/classification_module/mapping.txt'
croppedimg='./data/e2e_pipeline/croppedImage/'
test='./data/e2e_pipeline/test.txt'
