#%% ========== ========== ========== ========== ========== ========== ========== ==========
'''
incremental voronoi, divide and conque
'''
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import time
import sys
sys.path.append('../')
sys.path.append('../analysis/')
import copy
import platform
import json
import pickle
import random

import torch
from scipy.stats import mode
#%%
OS = platform.release()
nd = platform.uname().node
if OS.startswith('5.15'):
    if nd == 'MedImgGroup':
        work_dir = r'/home/chunweim/ssd/project/'
    elif nd == 'horsepurve-ThinkPad-T460s':
        work_dir = r'/home/horsepurve/Documents/Development/'
    elif nd == 'biomedimglab':
        work_dir = r'/media/SSD1/chunweim/'
    else:
        print('ERROR')
elif OS.startswith('5.3'):
    work_dir = r'/home/chunweima/usr/'
elif OS.startswith('3'):
    work_dir = r'/usr/home/chunma/dev/'
else:
    print('ERROR')
work_dir += r'pass/MNIST'
print('current dir:', work_dir, '@', nd)
os.chdir(work_dir)

#%%
from ResNet import resnet18_cbam

data_name     = 'CIFAR100' # 'Subset' # 'Tiny' # 'CIFAR100' # 'MNIST'
total_nc      = 100 # 200 # 100 # 10
fg_nc         = 50 # 100 # 40 # 50 # 4
task_num      = 5 # 20 # 5 # 10 # 2

# constant params
class_set     = list(range(total_nc))
ROTATION      = True
EMB2D         = False
DESIGN        = True
virtual_class = 4*total_nc if ROTATION else total_nc
suffix        = 'voro' # 'prob' 'voro'
assert (total_nc - fg_nc)%task_num == 0
task_size     = int((total_nc - fg_nc)/task_num)
file_name     = data_name+'_'+str(fg_nc)+'_'+str(task_num)+'x'+str(task_size)+suffix

print(file_name)

brother       = False # True False
#%% load model
save_path = '../checkpoints/'
path      = save_path + file_name + '/'
filename  = path + '%d_model.pkl' % (fg_nc+0) # +5
model     = torch.load(filename)
model.eval()
print('> loaded base phase model from:', filename)
#%% 
from voro_helper import num_param, t2n, \
                        load_feat, sanity_check, \
                        id2emb_, num_samp, merge_sc, funbtch

num_param(model)

# voronoi centers:
vcenter = t2n(model.module.classifier.weight.data / 2.)
print(vcenter.shape)
#%% ========== ========== ========== ========== ========== ========== ========== ==========
# test & train feature
'''
CIFAR:
    'CIFAR100_50_5x10_test.pkl'
    'CIFAR100_50_5x10_train.pkl'
    'CIFAR100_40_20x3_test.pkl'
    'CIFAR100_40_20x3_train.pkl'
more CIFAR:
    'CIFAR100_40_12x5_test.pkl'
    'CIFAR100_40_12x5_train.pkl'
    'CIFAR100_30_14x5_test.pkl'
    'CIFAR100_30_14x5_train.pkl'
    'CIFAR100_20_16x5_test.pkl'
    'CIFAR100_20_16x5_train.pkl'
    'CIFAR100_10_18x5_test.pkl'
    'CIFAR100_10_18x5_train.pkl'
Tiny:
    'Tiny_100_20x5_test.pkl'
    'Tiny_100_20x5_train.pkl'
TinyL3:
    'Tiny_100_5x20_test.pkl'
    'Tiny_100_5x20_train.pkl'
Subset:
    'Subset_50_10x5_test.pkl'
    'Subset_50_10x5_train.pkl'
'''
ffile = '../embedding/prob/' + 'CIFAR100_50_5x10_test.pkl' # prob probL3
data_phas, last_tsk = load_feat(ffile)

labs, embeds, outputs = data_phas[last_tsk][last_tsk]
false_id, pred_ = sanity_check(labs, embeds, outputs)
#%%
ffile = '../embedding/prob/' + 'CIFAR100_50_5x10_train.pkl' # prob probL3
data_phas, last_tsk = load_feat(ffile)

labst, embedst, outputst = data_phas[last_tsk][last_tsk]
false_id, pred_ = sanity_check(labst, embedst, outputst)
#%%
embs  = id2emb_(labs,  embeds,  brother=brother)
embst = id2emb_(labst, embedst, brother=brother)

sam_max, sam_min = num_samp(embst)
print('> loading data done!')
#%% random fill
for k,v in embst.items():
    if v.shape[0] < sam_max:
        rid = np.random.choice(v.shape[0], sam_max-v.shape[0], replace=False)
        print('filling', sam_max-v.shape[0], 'samples')
        embst[k] = np.vstack((v, v[rid]))
#%% for MNIST only
sam_max, sam_min = num_samp(embs)
for k,v in embs.items():
    if v.shape[0] < sam_max:
        rid = np.random.choice(v.shape[0], sam_max-v.shape[0], replace=False)
        print('filling', sam_max-v.shape[0], 'samples')
        embs[k] = np.vstack((v, v[rid]))
#%% 
from voro_helper import get_task_data, feat_trans, NC_o, tsk_cls, fun, acc_brother, forgetting

i = task_num
support_data_, support_label_, query_data_, query_label_, n_ways_, n_shot_, _ = \
    get_task_data(embst, embs, i, fg_nc, class_set, task_size, brother=brother)

#%% ========== ========== ========== ========== ========== ========== ========== ==========
be        = 1. # 1.
l2        = False # False True
protots   = np.mean(feat_trans(support_data_, beta=be, l2=l2), axis=1)

#%% have a first look
i = 0
support_data, support_label, query_data, query_label, n_ways, n_shot, classes = \
    get_task_data(embst, embs, i, fg_nc, class_set, task_size, brother=brother)
acc, _, nn_pre  = NC_o(support_data, support_label, 
                       query_data, query_label,
                       n_ways, n_shot, 
                       beta = be, l2 = l2, 
                       given_cen = protots[classes], given_only = True,
                       brother = brother)

#%% do it (NN) all together [see local phases]
local    = False # True | False
nn_acc   = []
savefile = path+'NN' if brother else None # NN | NNp | NN3 | only saved NN results!
savefil2 = path+'NN3'
forget   = True # True False # calculate forgetting
class_ladder = []
a        = []
for i in range(task_num + 1):
    support_data, support_label, query_data, query_label, n_ways, n_shot, classes = \
        get_task_data(embst, embs, i, fg_nc, class_set, task_size,
                      local = local, brother = brother) # False
    if forget:
        class_ladder.append(classes)
    acc, _, nn_pre  = NC_o(support_data, support_label, 
                           query_data, query_label,
                           n_ways, n_shot, 
                           beta = be, l2 = l2, 
                           given_cen = protots[classes] if local else protots[:classes[-1]+1], 
                           given_only = True,
                           brother = brother,
                           savefile  = savefile+'_'+str(i) if brother else None,
                           savefile2 = None, # savefil2+'_'+str(i), # None, # 
                           weights = [0.7, 0.3],
                           class_ladder = class_ladder,
                           acc_all = a
                           )
    nn_acc.append(acc)
    # if all classes up2now: local = False, and protots[:classes[-1]+1]
print('local:', local, '| beta:', be, '| l2:', l2)
if brother:
    for accs in nn_acc:
        print('\t'.join([str(i) for i in accs]))
else:
    print('\n'.join([str(i) for i in nn_acc]))
if forget:
    avgf = forgetting(a, task_num)
    print(avgf)
#%% see all logistic predictions
def predict(model, x, voro=True):
    x = torch.from_numpy(x.astype(np.float32)).cuda()
    with torch.no_grad():
        predi  = model.module.classifier(x)
    if voro:
        predi += - torch.sum((model.module.classifier.weight/2.)**2, dim=1)
    return predi
        
classes_all = []
test_share  = []
paccS       = []
for i in range(task_num + 1):
    support_data, support_label, query_data, query_label, n_ways, n_shot, classes = \
        get_task_data(embst, embs, i, fg_nc, class_set, task_size,
                      local = True, brother = brother)
    # load model:
    if brother:
        cls_01 = classes[-1]+1
        assert cls_01 % 4 == 0
        filename  = path + '%d_model.pkl' % (int(cls_01/4)) 
    else:
        filename  = path + '%d_model.pkl' % (classes[-1]+1) 
    model     = torch.load(filename)
    model.eval()
    print('> loaded base phase model from:', filename)
    num_param(model)
    # do prediction:
    min_cls = min(classes)
    if brother:
        pred = predict(model, query_data)
    else:
        pred = predict(model, query_data)[:,::4]
    pmax = torch.max(pred, dim=1)[1]
    pacc = np.mean(t2n(pmax) == query_label - min_cls)
    print(pacc)
    if brother:
        accs = acc_brother(t2n(pred), t2n(pmax), query_label - min_cls, forLG=True)
        pacc = [pacc] + accs
    paccS.append(pacc)
    classes_all.append(classes)
    test_share.append(len(query_label))
print('all logistic locally:')
if brother:
    for accs in paccS:
        print('\t'.join([str(i) for i in accs]))
else:
    print('\n'.join([str(i) for i in paccS]))
#%% prepare all prediction in each clique
for i in range(task_num + 1):
    support_data, support_label, query_data, query_label, n_ways, n_shot, classes = \
        get_task_data(embst, embs, i, fg_nc, class_set, task_size,
                      local = True, brother = brother)
    pred_all = []
    for classes in classes_all:
        # load model:
        if brother:
            cls_01 = classes[-1]+1
            assert cls_01 % 4 == 0
            filename  = path + '%d_model.pkl' % (int(cls_01/4)) 
        else:
            filename  = path + '%d_model.pkl' % (classes[-1]+1) 
        model     = torch.load(filename)
        model.eval()
        print('> loaded base phase model from:', filename)
        num_param(model)
        # do prediction:
        min_cls = min(classes)
        if brother:
            pred = predict(model, query_data)
        else:
            pred = predict(model, query_data)[:,::4]
        pmax = torch.max(pred, dim=1)[1]
        # pacc = np.mean(t2n(pmax) == query_label - min_cls)
        # print(pacc)
        if brother:
            labs_01 = query_label[::4]//4
            acc_, sc_pre = merge_sc(t2n(pred), labs_01, 
                                    mth='softmax', forLG=True, return_pre=True)
            min_cls = int(min_cls / 4)
            pred_all.append( (sc_pre + min_cls).tolist() )
        else:
            pred_all.append( (t2n(pmax) + min_cls).tolist() )
    pred_all = np.array(pred_all)
    print(pred_all.shape)
    
    if i == 0:
        pred_in_clique = copy.deepcopy(pred_all)
    else:
        pred_in_clique = np.hstack((pred_in_clique, pred_all))

# check it:
# plt.plot(pred_all[:,:100].T)
#%% clique-clique merging
print('cliques by test samples:', pred_in_clique.shape)

# compute prototypes
be        = 0.6 # 1.
l2        = True # False True
bias      = 0 # 0.05
protots   = np.mean(feat_trans(support_data_, beta=be, l2=l2, bias=bias), axis=1)
query_pts = feat_trans(query_data_, beta=be, l2=l2, bias=bias) 
# acenss    = feat_trans(acens, beta=be, l2=l2, bias=bias) 
if brother:
    dist      = funbtch(query_pts, protots) # protots | acenss
else:
    dist      = fun(query_pts, protots) # protots | acenss

test_splt = [sum(test_share[:t+1]) for t in range(len(test_share))]
#%% assign point-to-prototype distance to 6 candidate
dist_in_clique = np.zeros_like(pred_in_clique, dtype=np.float32)
for c_id, clique in enumerate(pred_in_clique):
    for s_id, samp in enumerate(clique):
        dist_in_clique[c_id][s_id] = dist[s_id][samp]
'''
pred_in_clique:
array([[ 0,  0,  0, ..., 26, 18, 30],
       [53, 57, 57, ..., 56, 55, 55],
       [62, 62, 62, ..., 61, 69, 67],
       [77, 77, 77, ..., 78, 78, 73],
       [83, 83, 83, ..., 86, 80, 87],
       [92, 92, 92, ..., 99, 99, 99]])
dist:
(10000, 100)
'''
#%% merge all
for i in range(task_num + 1):
    clique_dist = dist_in_clique[:i+1,:test_splt[i]]
    clique_pred = np.argmin(clique_dist, axis=0)
    merge_pred  = [pred_in_clique[cli,s_id] for s_id, cli in enumerate(clique_pred)]
    merge_acc   = np.mean(merge_pred == query_label_[:test_splt[i]])
    print(merge_acc)
print('DaC:', 'beta:', be, '| l2:', l2)
#%% ========== ========== ========== ========== ========== ========== ========== ==========
# ~~~~~ ~~~~~
# brother merging
# ~~~~~ ~~~~~

#%%
# for neo-VD
be        = 0.7 # 1.
l2        = True # False True
bias      = 0 # 0.05

#%%
print(acens.min())
acenss    = feat_trans(acens, beta=be, l2=l2, bias=bias) 

#%% 
# ~~~~~ ~~~~~ ~~~~~ ~~~~~
# merging, but advised by candidates
# ~~~~~ ~~~~~ ~~~~~ ~~~~~

test_splt = [sum(test_share[:t+1]) for t in range(len(test_share))]
local     = False # True | False
nn_acc    = []
savefile  = path+'NN' if brother else None # NN | NNp
savefil2  = path+'NN3'
centros   = protots # protots | acenss (not good)
for i in range(task_num + 1):
    candida = pred_in_clique[:i+1, :int(test_splt[i] / 4)]
    print(candida.shape)
    
    support_data, support_label, query_data, query_label, n_ways, n_shot, classes = \
        get_task_data(embst, embs, i, fg_nc, class_set, task_size,
                      local = local, brother = brother) # False
    acc, _, nn_pre  = NC_o(support_data, support_label, 
                           query_data, query_label,
                           n_ways, n_shot, 
                           beta = be, l2 = l2, 
                           given_cen = centros[classes] if local else centros[:classes[-1]+1], 
                           given_only = True,
                           brother = brother,
                           candida = candida, # candida | []
                           truth   = False, # False | True
                           savefile  = savefile+'_'+str(i), 
                           savefile2 = savefil2+'_'+str(i), # None, # 
                           weights = [0.7, 0.3] 
                           )
    nn_acc.append(acc)
    # if all classes up2now: local = False, and protots[:classes[-1]+1]
    # break
print('local:', local, '| beta:', be, '| l2:', l2)
if brother:
    for accs in nn_acc:
        print('\t'.join([str(i) for i in accs]))
else:
    print('\n'.join([str(i) for i in nn_acc]))

#%% deprecated below
print('@ this is some reason here, think about this @') # voting for brother: problematic
i = 1

candida = pred_in_clique[:i+1, :int(test_splt[i] / 4)]
print(candida.shape)

dist_in_clique = np.zeros((candida.shape[1], 4, candida.shape[0]*4))
print(dist_in_clique.shape)
for s_id in range(candida.shape[1]):
    for c_id in range(candida.shape[0]):
        _lab_ = candida[c_id,s_id]
        _lab_ = np.arange(_lab_, _lab_+4)
        # print(s_id, c_id, _lab_)
        dist_in_clique[s_id][:,c_id*4:c_id*4+4] = dist[s_id*4:s_id*4+4, _lab_]
print(dist_in_clique.shape)
pred_4  = np.argmin(dist_in_clique, axis = -1)
pred_4  = pred_4 // 4

pred_41 = np.array([mode(li)[0][0] for li in pred_4])
pred_41 = candida[pred_41,np.arange(candida.shape[1])]

labs_01 = query_label_[:test_splt[i]] 
labs_01 = labs_01[::4] // 4
acc_vot = np.mean(pred_41 == labs_01)
print(acc_vot)

#%%
#%% 
#%%
#%% 
#%% 
#%% 
#%%
#%% 
#%%
#%% 
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%% 
