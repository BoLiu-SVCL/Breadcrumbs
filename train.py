import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import tqdm
from collections import deque
import math
import numpy as np

from models.model import ResNet18feat, LinearClassifier
from data.dataloader import load_data
from data.ClassAwareSampler import get_sampler


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

n_class = 1000
n_channel = 512
batch_size = 512
n_epoch = 200

data_root = '/gpu7_ssd/liubo/ImageNet'
sampler_dic = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 4, 'sampler':get_sampler()}
train_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=16, test_open=False, shuffle=True)
train_plain_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=batch_size, sampler_dic=None, num_workers=16, test_open=False, shuffle=True)

is_ready = False
nb = math.ceil(len(train_plain_loader)*batch_size/n_class)

device = 'cuda:0'

feat_model = ResNet18feat()
cls_model_r = LinearClassifier(in_features=n_channel, out_features=n_class)
cls_model_c = LinearClassifier(in_features=n_channel, out_features=n_class)
feat_model.to(device)
cls_model_r.to(device)
cls_model_c.to(device)
feat_model = nn.DataParallel(feat_model)
cls_model_r = nn.DataParallel(cls_model_r)
cls_model_c = nn.DataParallel(cls_model_c)

sm_loss = nn.CrossEntropyLoss()

optim_params_list = [{'params': feat_model.parameters(), 'lr': 0.2, 'momentum': 0.9, 'weight_decay': 0.0005},
                     {'params': cls_model_r.parameters(), 'lr': 0.2, 'momentum': 0.9, 'weight_decay': 0.0005}]
optimizer = optim.SGD(optim_params_list)
optimizer_cls = optim.SGD(params=cls_model_c.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0.0)
scheduler_cls = optim.lr_scheduler.CosineAnnealingLR(optimizer_cls, T_max=n_epoch, eta_min=0.0)

training_labels = np.array(train_plain_loader.dataset.labels).astype(int)
train_class_count = []
for c in range(n_class):
    train_class_count.append(len(training_labels[training_labels == c]))
ql_list = []
feats_list = []
for c in range(n_class):
    feats_list.append(deque())
    ql_list.append(math.ceil(nb/train_class_count[c]))

e_train_loader = enumerate(train_loader, 0)
for epoch in range(n_epoch):
    feat_model.train()
    cls_model_r.train()
    cls_model_c.train()
    loss_all = 0.0
    current_feats = torch.empty((0, n_channel)).to(device)
    current_labels = torch.empty(0, dtype=torch.long).to(device)
    if is_ready:
        train_feats = torch.empty((0, n_channel)).to(device)
        train_labels = torch.empty(0, dtype=torch.long).to(device)
        for c in range(n_class):
            c_feats = torch.empty((0, n_channel)).to(device)
            for e in range(ql_list[c]):
                feats = feats_list[c][-1-e][0] - feats_list[c][-1-e][1]
                c_feats = torch.cat((c_feats, feats))
            c_feats = c_feats[:nb, :] + feats_list[c][-1][1]
            c_labels = c * torch.ones(nb, dtype=torch.long).to(device)
            train_feats = torch.cat((train_feats, c_feats))
            train_labels = torch.cat((train_labels, c_labels))
        train_idx = torch.randperm(train_feats.size(0))[:batch_size*len(train_plain_loader)].to(device)
        train_feats = torch.index_select(train_feats, 0, train_idx)
        train_labels = train_labels[train_idx]
    for i, (images, labels, paths) in tqdm(enumerate(train_plain_loader, 0)):
        images, labels = images.to(device), labels.to(device)
        if is_ready:
            feat_map_cls = train_feats[i*batch_size:(i+1)*batch_size, :]
            labels_CB = train_labels[i*batch_size:(i+1)*batch_size]
        else:
            j, (images_CB, labels_CB, paths_CB) = next(e_train_loader)
            if j == len(train_loader) - 1:
                e_train_loader = enumerate(train_loader)
            images_CB, labels_CB = images_CB.to(device), labels_CB.to(device)

            feat_map_cls = feat_model(images_CB).detach()
        logits_cls = cls_model_c(feat_map_cls)
        loss_cls = sm_loss(logits_cls, labels_CB)
        optimizer_cls.zero_grad()
        loss_cls.backward()
        optimizer_cls.step()
        loss_all += loss_cls.item()

        feat_map = feat_model(images)
        logits = cls_model_r(feat_map)
        loss_r = sm_loss(logits, labels)
        optimizer.zero_grad()
        loss_r.backward()
        optimizer.step()
        loss_all += loss_r.item()

        current_feats = torch.cat((current_feats, feat_map.detach()))
        current_labels = torch.cat((current_labels, labels.detach()))
    is_ready = True
    for c in range(n_class):
        c_idx = torch.where(current_labels == c)[0]
        c_feats = torch.index_select(current_feats, 0, c_idx)
        c_avg = c_feats.mean(dim=0, keepdim=True)
        if len(feats_list[c]) >= ql_list[c]:
            feats_list[c].popleft()
        feats_list[c].append((c_feats, c_avg))
        if len(feats_list[c]) < ql_list[c]:
            is_ready = False

    loss_all /= len(train_plain_loader)
    print('Epoch[{}, {}] loss: {:.4f}'.format(epoch, n_epoch, loss_all))

    scheduler_cls.step()
    scheduler.step()

torch.save({'feat': feat_model.state_dict(), 'cls_r': cls_model_r.state_dict(), 'cls_c': cls_model_c.state_dict()}, 'Breadcrumb.pth')
