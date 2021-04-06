
## This classification require a trained model of all 10 classes.
# Best practice : -s 3 -c 0.010000 -q : train acc: 98.4214, test acc: 93.7225
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import torch
import sys
sys.path.append('.')
sys.path.append('./src')
import src.network_torch as network_torch
from src.model_point_torch import GPointNet
import multiprocessing as mp
import matplotlib.pyplot as plt

import os

import math
from utils.eulerangles import euler2mat
import svm
from liblinear.python.liblinearutil import *
from liblinear.python.liblinear import *
from tqdm import tqdm
def train_svm(train_feature, train_labels, test_features, test_labels):

    # train SVM
    print('Begin to train SVM .........')
    train_feature = train_feature.tolist()
    train_labels = train_labels.tolist()
    test_features = test_features.tolist()
    test_labels = test_labels.tolist()
    
    prob = problem(train_labels, train_feature)
    for t in range(1):
        for c in [0.001, 0.005, 0.01, 0.02]:
            para = '-s 2 -c %f -q' % (c)
            param = parameter(para)
            svm_model = train(prob, param)
            _, train_acc, _ = predict(train_labels, train_feature, svm_model)
            _, test_acc, _ = predict(test_labels, test_features, svm_model)
            print('%s : train acc: %.4f, test acc: \033[92m%.4f\033[0m' % (para,train_acc[0], test_acc[0]))

def train_svm_par(train_feature, train_labels, test_features, test_labels):

    # train SVM
    print('Begin to train SVM .........')
    train_feature = train_feature.tolist()
    train_labels = train_labels.tolist()
    test_features = test_features.tolist()
    test_labels = test_labels.tolist()
    
    pool = mp.Pool(processes=16)
    para_list = []
    for s in range(6):
        for c in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 2, 4, 6, 10]:
            para_list.append('-s %d -c %f -q' % (s, c))

    pool_res = [pool.apply_async(par_svm, args=(para,train_feature, train_labels, test_features, test_labels)) 
                for para in para_list]
    results = [res.get(timeout=1000) for res in pool_res]
    for r in results: 
        try: 
            print('%s : train acc: %.4f, test acc: \033[92m%.4f\033[0m' % (r[0],r[1],r[2]))
        except: 
            print('timeout.')

def par_svm(para, train_feature, train_labels, test_features, test_labels):

    prob = problem(train_labels, train_feature)
    param = parameter(para)
    svm_model = train(prob, param)
    _, train_acc, _ = predict(train_labels, train_feature, svm_model)
    _, test_acc, _ = predict(test_labels, test_features, svm_model)
    print('%s : train acc: %.4f, test acc: \033[92m%.4f\033[0m' % (para,train_acc[0], test_acc[0]))
    return (para, train_acc[0], test_acc[0])

def robust_test(config, model, train_features, train_labels, test_features, test_labels, test_data): 

    font_size = 18
    # Missing test
    param = parameter('-s 3 -c 0.010000 -q')
    ori_train = train_features.tolist()
    train_labels = train_labels.tolist()
    ori_test = test_features.tolist()
    test_labels = test_labels.tolist()
    prob = problem(train_labels, ori_train)
    svm_model = train(prob, param)
    _, train_acc, _ = predict(train_labels, ori_train, svm_model)
    _, test_acc, _ = predict(test_labels, ori_test, svm_model)
    print('Oringinal : train acc: %.4f, test acc: \033[92m%.4f\033[0m' % (train_acc[0], test_acc[0]))

    percentage = np.array([1,2,3,4,6,8,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
    pnt_batch = [int(test_data.shape[1] * pe / 100) for pe in percentage]
    test_feature_all = extract_feature(config, model, test_data, robust=pnt_batch)
    test_acc = []
    for percent, num_pnt in zip(percentage, pnt_batch):
        print("Current testing percent %d, %d points" % (percent, num_pnt))
        tr = test_feature_all[num_pnt].tolist()
        _, acc, _ = predict(test_labels, tr, svm_model)
        print('Robust missing: %d %% : test acc: \033[92m%.4f\033[0m' % (percent, acc[0]))
        test_acc.append(acc[0])
    plt.clf()
    plt.xlabel("Missing Point Ratio", fontsize=font_size)
    plt.ylabel("Testing Accuracy (%)", fontsize=font_size)
    plt.plot(1 - percentage / 100, test_acc, 'bs')
    plt.plot(1 - percentage / 100, test_acc, 'b', linewidth=3)
    plt.savefig("output/classification/robustness_test_missing.png")
    plt.savefig("output/classification/robustness_test_missing.eps")

    # Add noise test 

    num_pnt = 2048
    percentage = np.array([1,2,3,4,6,8,10,15,20,25,30,35,40,45,50,60,70,80,90,100])
    pnt_batch = [int(test_data.shape[1] * pe / 100) for pe in percentage]
    test_acc = []
    x_min, x_max = test_data.min(), test_data.max()
    noise_data = np.random.normal(0, 0.3, [test_data.shape[0], num_pnt, 3]).astype(np.float32)
    test_feature_noise = extract_feature(config, model, noise_data, robust=percentage)
    for percent, num_pnt in zip(percentage, pnt_batch):
        print("Current testing to add percent %d, %d points" % (percent, num_pnt))
        tr = np.maximum(test_features, test_feature_noise[percent])
        tr = tr.tolist()
        _, acc, _ = predict(test_labels, tr, svm_model)
        print('Add noise point %d%% : test acc: \033[92m%.4f\033[0m' % (percent, acc[0]))
        test_acc.append(acc[0])
    plt.clf()
    plt.xlabel("Added Point Ratio", fontsize=font_size)
    plt.ylabel("Testing Accuracy (%)", fontsize=font_size)
    plt.plot(percentage / 100, test_acc, 'bs')
    plt.plot(percentage / 100, test_acc, 'b', linewidth=3)
    plt.savefig("output/classification/robustness_test_add.png")
    plt.savefig("output/classification/robustness_test_add.eps")

    # Add noise

    std_batch = [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
    test_acc = []
    for std in std_batch:
        print("Current testing to add noise of std=%.3f" % std)
        noise_data = np.random.normal(0, std, [test_data.shape[0], test_data.shape[1], 3]).astype(np.float32)
        test_data_noise = test_data + noise_data
        test_feature_noise = extract_feature(config, model, test_data_noise)
        tr = test_feature_noise.tolist()
        _, acc, _ = predict(test_labels, tr, svm_model)
        print('Add gaussian noise N(0, %.4f): test acc: \033[92m%.4f\033[0m' % (std, acc[0]))
        test_acc.append(acc[0])
    plt.clf()
    plt.tight_layout()
    labelss = plt.xlabel("Standard Deviation for Adding Noise", fontsize=font_size)
    plt.ylabel("Testing Accuracy (%)", fontsize=font_size)
    plt.plot(std_batch, test_acc, 'bs')
    plt.plot(std_batch, test_acc, 'b', linewidth=3)
    plt.xscale('log')
    plt.savefig("output/classification/robustness_test_noise.png", bbox_extra_artists=(labelss))
    plt.savefig("output/classification/robustness_test_noise.eps", bbox_extra_artists=(labelss))

    return

def parse_config():

    parser = argparse.ArgumentParser()

    # Point cloud related
    parser.add_argument('-hidden_size', type=list, default=[[64,128,256,512,1024], [512,256,64]])
    parser.add_argument('-net_type', type=str, default="default_big")
    parser.add_argument('-num_point', type=int, default=2048)
    parser.add_argument('-point_dim', type=int, default=3)
    parser.add_argument('-argment_mode', type=int, default=0)
    parser.add_argument('-argment_noise', type=float, default=0.01)
    parser.add_argument('-batch_norm', type=str, default="ln", help='BatchNorm(bn) / LayerNorm(ln) / InstanceNorm(in) / None')
    parser.add_argument('-activate_eval', type=int, default=0)
    parser.add_argument('-mode', type=str, default="max")
    parser.add_argument('-tpp', type=int, default=1)
    parser.add_argument('-robust_test', action='store_true')
    
    # EBM related
    parser.add_argument('-batch_size', type=int, default=32, help='')
    parser.add_argument('-activation', type=str, default="ReLU", help='')
    parser.add_argument('-seed', type=int, default=666, help='')
    parser.add_argument('-data_size', type=int, default=10000)
    parser.add_argument('-cuda', type=str, default="0", help='')
    parser.add_argument('-checkpoint_path', type=str, default="output/pytorch/modelnet10_default_big_nlr5e4_epo_1200.ckpt")
    parser.add_argument('-output_dir', type=str, default="output/classification")
    return parser.parse_args()

def load_model(config):

    model = GPointNet(config, getattr(network_torch, "energy_point_" + config.net_type)(config))
    if config.checkpoint_path[-4:] != "ckpt":
        path = "output/pytorch/" + config.checkpoint_path + "/lightning_logs/version_%d"
        for i in range(10, -1, -1):
            if os.path.isdir(path % i):
                path = path % i 
                break 
        path += "/checkpoints"
        for file in os.listdir(path):
            if file[-4:] == "ckpt":
                break 
        config.checkpoint_path = path + "/" + file 
        print("found: %s" % config.checkpoint_path)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location="cuda:0")["state_dict"])

    # Change layerNorm 
    loc = model.energy_net.local
    for layer in loc: 
        if type(layer) is torch.nn.LayerNorm: 
            layer.weight = torch.nn.Parameter(layer.weight.mean().repeat(2048))
            layer.bias = torch.nn.Parameter(layer.bias.mean().repeat(2048))
            layer.normalized_shape = tuple((2048,))
    print("Loaded: %s" % config.checkpoint_path)
    model.cuda()    
    return model 

def extract_feature(config, model, data, robust=None):

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.swapaxes(data, 1, 2)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    feature = [] if robust is None else {k: [] for k in robust}
    idx = np.random.permutation(data.shape[1])
    for i, batch in enumerate(tqdm(loader)): 
        with torch.no_grad():
            fea = model.energy_net(batch[0].cuda(), out_local=True).cpu().data.numpy()
        if robust is None: 
            fea = feature.append(np.mean(fea, -1) if config.mode == "mean" else np.max(fea, -1))
        else: 
            for k in robust: 
                feature[k].append(np.mean(fea[..., idx[:k]], -1) if config.mode == "mean" else np.max(fea[..., idx[:k]], -1))
    if robust is None:
        feature = np.concatenate(feature)
    else: 
        for k in robust:
            feature[k] = np.concatenate(feature[k])
    return feature

def main():

    opt = parse_config()
    if opt.net_type == "default_residual":
        opt.net_type = "default"
        opt.hidden_size = [[64,64,64,128,128,128,256,256,256,512,1024], [512,256,256,256,64,64,64]]
    elif opt.net_type == "default_big":
        opt.net_type = "default"
        opt.hidden_size = [[64,128,256,512,1024,2048], [1024,512,256,64]]
    opt.swap_axis = True
    opt.do_evaluation = 0
    np.random.seed(opt.seed)
    if opt.cuda==1: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Prepare training data
    
    categories = ["toilet", "table", "sofa", "night_stand", "monitor", "dresser", "desk", "chair", "bed", "bathtub"]
    train_data, train_labels, train_idx = [], [], []
    for idx, cat in enumerate(categories):
        d = np.load('data/%s_train.npy' % cat)
        train_data.append(d)
        train_labels.append(np.ones(d.shape[0]) * idx)
        train_idx.append(np.array([idx * 10000 + i for i in range(d.shape[0])]))
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    train_idx = np.concatenate(train_idx)
    if opt.argment_mode:
        print("Argment data by rotating through x axis. %d Data" % len(train_data))
        out_data = []
        for xrot in (np.arange(-1, 1, 0.25) * math.pi):
            M = euler2mat(0, xrot, 0)
            out_data.append(np.dot(train_data, M.transpose()))
        train_data = np.concatenate(out_data).astype(np.float32)
        train_labels = train_labels.repeat(8, axis=0)
        train_idx = train_idx.repeat(8, axis=0)
        print("Argmented. %d Data" % len(train_data))

    idx = np.random.permutation(train_data.shape[0])
    train_data = train_data[idx, :opt.num_point]
    train_labels = train_labels[idx]
    train_idx = train_idx[idx]
    np.save("%s/train_idx.npy" % opt.output_dir, idx)

    test_data, test_labels = [], []
    for idx, cat in enumerate(categories):
        d = np.load('data/%s_test.npy' % cat)
        test_data.append(d)
        test_labels.append(np.ones(d.shape[0]) * idx)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    idx = np.random.permutation(test_data.shape[0])
    test_data = test_data[idx, :opt.num_point]
    test_labels = test_labels[idx]
    np.save("%s/train_labels.npy" % opt.output_dir, train_labels)
    np.save("%s/test_labels.npy" % opt.output_dir, test_labels)

    max_val = train_data.max()
    min_val = train_data.min()
    train_data = ((train_data - min_val) / (max_val - min_val)) * 2 - 1
    test_data = ((test_data - min_val) / (max_val - min_val)) * 2 - 1


    if opt.tpp and os.path.exists(opt.output_dir + "/train_feature.npy"):
        train_feature = np.load(opt.output_dir + "/train_feature.npy")
        test_feature = np.load(opt.output_dir + "/test_feature.npy")
        print("pretrained data loaded.")
        print(train_feature.shape)
        if opt.robust_test:
            model = load_model(opt)
    else: 
        model = load_model(opt)
        train_feature = extract_feature(opt, model, train_data)
        test_feature = extract_feature(opt, model, test_data)
        np.save(opt.output_dir + "/train_feature.npy", train_feature)
        np.save(opt.output_dir + "/test_feature.npy", test_feature)

    # voxel_mean = train_feature.mean()
    # voxel_var = np.sqrt(train_feature.var())
    # voxel_mean = np.mean(train_feature, 0, keepdims=True)
    # voxel_var = np.sqrt(np.var(train_feature, 0, keepdims=True))
    # train_feature = (train_feature - voxel_mean) / voxel_var
    # test_feature = (test_feature - voxel_mean) / voxel_var
    if opt.robust_test: 
        robust_test(opt, model, train_feature, train_labels, test_feature, test_labels, test_data)
    else: 
        train_svm_par(train_feature, train_labels, test_feature, test_labels)

if __name__ == '__main__':
    main()