# 2021.03.01 

# Logistic
import tqdm 
import argparse 
import traceback 
import datetime
import json
import os 
import sys
sys.path.append('.')
sys.path.append('./src')
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

# Main
import torch 
import pytorch_lightning as pl 
from utils import data_util
from utils import util_torch
import numpy as np
from src.model_point_torch import GPointNet
import src.network_torch as network_torch
from metrics.evaluation_metrics import one_for_one_EMD_CD_
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product, combinations

def parse_config():

    parser = argparse.ArgumentParser()

    # Point cloud related
    
    parser.add_argument('-rec_step_size', type=float, default=0.2, help='')
    parser.add_argument('-rec_step', type=int, default=64, help='')
    parser.add_argument('-batch_size', type=int, default=128, help='')
    parser.add_argument('-category', type=str, default="modelnet")
    parser.add_argument('-output_dir', type=str, default="default")
    parser.add_argument('-synthesis', action='store_true')
    parser.add_argument('-evaluation', action='store_true')
    parser.add_argument('-reconstruction', action='store_true')
    parser.add_argument('-intepolation', action='store_true')
    parser.add_argument('-visualize_layer', action='store_true')
    parser.add_argument('-cuda', type=str, default="0", help='')
    parser.add_argument('-max_num', type=int, default=4, help='')
    parser.add_argument('-visualize_mode', type=int, default=3)
    parser.add_argument('-checkpoint_path', type=str, default="output/pytorch/modelnet10_default_big_nlr5e4_epo_1200.ckpt")
    
    return parser.parse_args()

def reconstruction(model, opt): 

    model.C.rec_step = opt.rec_step
    model.C.rec_step_size = opt.rec_step_size
    pcs = np.load(opt.data_path)[:opt.max_num]
    max_val = pcs.max()
    min_val = pcs.min()
    pcs = ((pcs - min_val) / (max_val - min_val)) * 2 - 1
    pcs = np.swapaxes(pcs, 1, 2)
    pcs = torch.from_numpy(pcs).cuda()
    idx_start = 0 
    latent_pcs, rec_pcs = [], []
    while idx_start < len(pcs):
        lat, rec, error = model(reconstruct_pcs=pcs[idx_start:idx_start+opt.batch_size])
        latent_pcs.append(lat.cpu().data.numpy())
        rec_pcs.append(rec.cpu().data.numpy())
        # if idx_start == 0: 
        #     ref = torch.swapaxes(pcs[idx_start:idx_start+opt.batch_size], 1, 2)
        #     print(opt.output_dir, error * 1000, one_for_one_EMD_CD_(rec.detach(), ref.detach()))
        #     rec = rec.cpu().data.numpy()
        #     rec = np.swapaxes(rec, 1, 2)
        #     lat = np.swapaxes(lat, 1, 2)
        #     util_torch.visualize(rec, num_rows=5, num_cols=5, mode=1, output_filename="%s/rec.png" % opt.output_dir)
            # if opt.evaluation:
        idx_start += opt.batch_size
    return np.concatenate(latent_pcs), np.concatenate(rec_pcs), error

def output_voxel(voxel, layer, i, cate, output_folder):

    r = [-0.2, voxel.shape[-1] + 0.2]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection='3d')
    plt.subplots_adjust(0,0,1,1,0,0)
    ax.axis('off')
    ax.voxels(voxel, facecolors=[0, 1, 1, 0.3], edgecolor='k', linewidth=0.14)
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=3, zorder=-1)
    ax.plot3D([r[0], r[1]], [r[0], r[0]], [r[1], r[1]], color="k", linewidth=3, zorder=10000000)
    ax.plot3D([r[1], r[1]], [r[0], r[1]], [r[1], r[1]], color="k", linewidth=3, zorder=10000000)
    ax.plot3D([r[1], r[1]], [r[0], r[0]], [r[0], r[1]], color="k", linewidth=3, zorder=10000000)
    plt.savefig(output_folder + "/%s_layer%d_%d.png" % (cate, layer, i))
    # plt.savefig(output_folder + "outout_voxel_%s_layer%d_%d.eps" % (cate, layer, i))
    plt.close()
    print("Printed voxel image of layer %d, %dth." %(layer, i))

def visualize_layer(model, opt):


    grid = []
    output_folder = "output/virsualize_layer/"
    resolution = 50
    
    # Change layerNorm 
    loc = model.energy_net.local
    for layer in loc: 
        if type(layer) is torch.nn.LayerNorm: 
            layer.weight = torch.nn.Parameter(layer.weight.mean().repeat(resolution**3))
            layer.bias = torch.nn.Parameter(layer.bias.mean().repeat(resolution**3))
            layer.normalized_shape = tuple((resolution**3,))
    print_amount = 64
    parallel = 16
    mode = "voxel"
    for x in np.linspace(-1, 1, resolution):
        for y in np.linspace(-1, 1, resolution):
            for z in np.linspace(-1, 1, resolution):
                grid.append([x, y, z])
    grid = np.array(grid)
    pcs = torch.from_numpy(grid[np.newaxis, ...]).float().cuda()
    pcs = pcs.transpose(1, 2)
    res = model.energy_net._output_all(pcs)
    res = [r.cpu().data.numpy() for r in res]
    print("Finish feature collecting......")
    
    pool = mp.Pool(processes=parallel) if parallel>1 else None
    t = np.linspace(-1, 1, resolution)
    idx = {}
    for i, tt in enumerate(t):
        idx[tt] = i

    for layer, curr_res in enumerate(res):

        curr_res = curr_res[0]
        print(curr_res.shape)
        print("Processing layer %d" % layer)
        sum_of_r = np.mean(curr_res, axis=-1)
        count_r = np.sum((curr_res>0), axis=-1)
        order = np.argsort(-sum_of_r)
        print([sum_of_r[order[k]] for k in range(32)])
        print([count_r[order[k]] for k in range(32)])
        voxel = np.zeros([curr_res.shape[0], len(t), len(t), len(t)])
        for i in range(curr_res.shape[0]): 
            for p, pnt in enumerate(grid):
                voxel[i, idx[pnt[0]], idx[pnt[1]], idx[pnt[2]]] = curr_res[order[i], p] > 0
        np.save(opt.output_dir + "/voxel_layer_%s_%d.npy" % (opt.category, layer), voxel)
        print("Saved layer %d" % layer)

        if pool is None:
            for i in range(print_amount):
                output_voxel(voxel[i], layer, i ,opt.category, opt.output_dir)
        else:
            pool_res = [pool.apply_async(output_voxel, args=(voxel[i], layer, i, opt.category, 
            opt.output_dir)) for i in range(print_amount)]
            result = [res.get(timeout=100) for res in pool_res]

def main(opt):

    if opt.cuda == "-1":
        opt.cuda = 1
    elif opt.cuda == "-2":
        opt.cuda = None
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    opt.data_path = "data/%s_train.npy" % opt.category
    opt.output_dir = opt.checkpoint_path[:-5] if opt.output_dir=="default" else opt.output_dir
    print(opt.output_dir)
    try:
        current_step = int(opt.checkpoint_path.split("_")[-1].split(".")[0])
    except: 
        current_step = 0
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    model = GPointNet.load_from_checkpoint(opt.checkpoint_path).cuda()

    if opt.visualize_layer: 
        visualize_layer(model, opt)
    
    if opt.synthesis: 

        syn_pcs = model(batch_size=min(model.C.batch_size, opt.max_num)).cpu().data.numpy()
        util_torch.visualize(syn_pcs[:20], num_rows=5, num_cols=4, output_filename="%s/syn_%d.png" % (opt.output_dir, current_step), mode=1)
        np.save("%s/syn_%d.npy" % (opt.output_dir, current_step), syn_pcs)

        if opt.evaluation:
            ref_pcs = np.load("data/%s_test.npy" % opt.category)
            logger = torch.utils.tensorboard.SummaryWriter("%s/lightning_logs/version_0" % opt.output_dir)
            logger.add_image("syn/re_black_syn", util_torch.visualize(syn_pcs, num_rows=5, num_cols=4, mode=1), current_step, dataformats='HW')
            res = util_torch.quantitative_analysis(syn_pcs, ref_pcs, model.C.batch_size, full=False)
            for k in res:
                if type(res[k]) is torch.Tensor:
                    res[k] = float(res[k].cpu().data)
            print("%30s|%4d|%.4f|%.4f|%.4f|%.4f|%.4f|" % (opt.output_dir[-30:], current_step, 
                    res['jsd'], res['mmd-CD'], res['mmd-EMD'], res['cov-CD'], res['cov-EMD']))
            for name, scalar in res.items():
                logger.add_scalar('syn/%s' % name, scalar, current_step)
            logger.flush()

    if opt.reconstruction: 

        latent_pcs, rec_pcs, error = reconstruction(model, opt)
        np.save("%s/rec_%d.npy" % (opt.output_dir, current_step), rec_pcs)
        np.save("%s/lat_%d.npy" % (opt.output_dir, current_step), latent_pcs)
        util_torch.visualize(rec_pcs[:20], num_rows=5, num_cols=4, output_filename="%s/rec_%d.png" % (opt.output_dir, current_step), mode=1)
        if opt.evaluation: 
            rec = torch.from_numpy(rec_pcs).cuda()
            ref = torch.from_numpy(np.load(opt.data_path)).cuda()
            cd, emd = one_for_one_EMD_CD_(rec, ref)
            logger = torch.utils.tensorboard.SummaryWriter("%s/lightning_logs/version_0" % opt.output_dir)
            logger.add_image("syn/reconstruct", util_torch.visualize(rec_pcs, num_rows=5, num_cols=4, mode=1), current_step, dataformats='HW')
            logger.add_scalar('syn/rec_CD', cd*100, current_step)
            logger.add_scalar('syn/rec_EMD', emd*10, current_step)
            logger.add_scalar('syn/rec_error', error, current_step)
            print("Reconstruction: CD:%.4f; EMD:%4f; error:%.4f" % (cd.cpu().data.numpy()*100, emd.cpu().data.numpy()*10, error))
            logger.flush()

    if opt.intepolation:

        if not os.path.exists("%s/lat_%d.npy" % (opt.output_dir, current_step)):
            print("Latent not found. Reloading...")
            latent_pcs, rec_pcs, _ = reconstruction(model, opt)
            np.save("%s/rec_%d.npy" % (opt.output_dir, current_step), rec_pcs)
            np.save("%s/lat_%d.npy" % (opt.output_dir, current_step), latent_pcs)
        else: 
            latent_pcs = np.load("%s/lat_%d.npy" % (opt.output_dir, current_step), latent_pcs)
        inte_idx = np.random.choice(len(latent_pcs), (2, opt.batch_size // 8), replace=True)
        pcs_left, pcs_right = latent_pcs[inte_idx[0]], latent_pcs[inte_idx[1]]
        pcs_inte = np.linspace(pcs_left, pcs_right, 8)
        pcs_inte = np.swapaxes(pcs_inte, 0, 1)
        out_pcs = model(torch.from_numpy(np.concatenate(pcs_inte)).cuda()).cpu().data.numpy() 
        out_pcs = np.swapaxes(out_pcs, 1, 2)
        util_torch.visualize(out_pcs, num_rows=16, num_cols=8, mode=1, output_filename="%s/intepolation.png" % opt.output_dir)
        np.save("%s/intepolation_%d.npy" % (opt.output_dir, current_step), out_pcs)
        np.save("%s/intepolation_idx_%d.npy" % (opt.output_dir, current_step), inte_idx)
        if opt.evaluation: 
            logger = torch.utils.tensorboard.SummaryWriter("%s/lightning_logs/version_0" % opt.output_dir)
            logger.add_image("syn/intepolation", util_torch.visualize(out_pcs, num_rows=16, num_cols=8, mode=1), dataformats='HW')

if __name__ == '__main__':

    opt = parse_config()
    main(opt)