import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from tqdm import tqdm

from models.model import ResNet18feat, LinearClassifier
from data.dataloader import load_data
from utils import mic_acc_cal, shot_acc


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

n_class = 1000
n_channel = 512
batch_size = 512

data_root = '/gpu7_ssd/liubo/ImageNet'
train_plain_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=batch_size, sampler_dic=None, num_workers=16, test_open=False, shuffle=True)
test_loader = load_data(data_root, dataset='ImageNet_LT', phase='test', batch_size=batch_size, sampler_dic=None, num_workers=16, test_open=False, shuffle=False)

device = 'cuda:0'

feat_model = ResNet18feat()
cls_model_c = LinearClassifier(in_features=n_channel, out_features=n_class)
feat_model.to(device)
cls_model_c.to(device)
feat_model = nn.DataParallel(feat_model)
cls_model_c = nn.DataParallel(cls_model_c)

state_dict = torch.load('Breadcrumb.pth')
feat_model.load_state_dict(state_dict['feat'])
cls_model_c.load_state_dict(state_dict['cls_c'])

feat_model.eval()
cls_model_c.eval()

total_logits = torch.empty((0, n_class)).to(device)
total_labels = torch.empty(0, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (images, labels, paths) in tqdm(enumerate(test_loader, 0)):
        images, labels = images.to(device), labels.to(device)
        feat_map = feat_model(images)
        logits = cls_model_c(feat_map)

        total_logits = torch.cat((total_logits, logits))
        total_labels = torch.cat((total_labels, labels))

probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

eval_acc_mic_top1 = mic_acc_cal(preds[total_labels != -1], total_labels[total_labels != -1])
many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds[total_labels != -1], total_labels[total_labels != -1], train_plain_loader)
# Top-1 accuracy and additional string
print_str = ['Evaluation_accuracy_micro_top1: %.3f'
             % (eval_acc_mic_top1),
             '\n',
             'Many_shot_accuracy_top1: %.3f'
             % (many_acc_top1),
             'Median_shot_accuracy_top1: %.3f'
             % (median_acc_top1),
             'Low_shot_accuracy_top1: %.3f'
             % (low_acc_top1),
             '\n']
print(print_str)
