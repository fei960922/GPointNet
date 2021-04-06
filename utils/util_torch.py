import json 
import traceback 
import functools
import requests
import torch
import numpy as np
import os
import sys
import time
sys.path.append('.')
sys.path.append('./src')
import matplotlib
matplotlib.use('Agg')
import h5py
import pickle
import math
import pytorch_lightning as pl
from utils.eulerangles import euler2mat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
try: 
    from metrics.evaluation_metrics import *
except: 
    print("Eval not available.")
import subprocess
import argparse
from PIL import Image


def visualize(point_clouds, num_rows=3, num_cols=3, output_filename=None, mode=0, idx=None):

    if point_clouds.shape[1] < 10: 
        point_clouds = np.swapaxes(point_clouds, 1, 2)
    num_clouds = len(point_clouds)
    # num_rows = min(num_rows, num_clouds // num_cols + 1)
    if mode == 0: 
        fig = plt.figure(figsize=(num_cols * 4, num_rows * 4))
        for i, pts in enumerate(point_clouds[:num_cols*num_rows]):
            if point_clouds.shape[2] == 3: 
                ax = plt.subplot(num_rows, num_cols, i+1, projection='3d')
                plt.subplots_adjust(0,0,1,1,0,0)
                #ax.axis('off')
                if idx is not None:
                    ax.set_title(str(idx[i]))
                ax.scatter(pts[:,0], pts[:,2], pts[:,1], marker='.', s=50, c=pts[:,2], cmap=plt.get_cmap('gist_rainbow'))
            else: 
                ax = plt.subplot(num_rows, num_cols, i+1)
                plt.subplots_adjust(0,0,1,1,0,0)
                # ax.axis('off')
                if idx is not None:
                    ax.set_title(str(idx[i]))
                ax.scatter(pts[:,1], -pts[:,0], marker='.', s=30)
        if output_filename is not None:
            plt.savefig(output_filename, bbox_inches='tight')
        return fig
    elif mode == 1: 
        row_imgs = []
        for ir in range(num_rows):
            col_imgs = []
            for ic in range(num_cols):
                idx = ir * num_cols + ic
                col_imgs.append(draw_point_cloud(point_clouds[idx], zrot=80 / 180.0 * np.pi,
                                    xrot=-45 / 180.0 * np.pi, yrot=-20 / 180.0 * np.pi)
                                if idx < point_clouds.shape[0] else np.zeros((500, 500)))
            row_imgs.append(np.concatenate(col_imgs, axis=1))
        im_array = np.concatenate(row_imgs, axis=0)
        img = Image.fromarray(np.uint8(im_array * 255.0))
        if output_filename is not None:
            img.save(output_filename)
        return np.array(img)
    elif mode == 3: 
        assert output_filename is not None, "in mode 3, output filename should not be None."
        if not os.path.exists(output_filename):
            os.makedirs(output_filename)
        for i, pts in enumerate(point_clouds):
            img = draw_point_cloud(point_clouds[i],
                                zrot=80 / 180.0 * np.pi,
                                xrot=-45 / 180.0 * np.pi,
                                yrot=-20 / 180.0 * np.pi)
            img = Image.fromarray(np.uint8(img * 255.0))
            img.save('%s/%d.png' % (output_filename, i))


class async_evaluation(object):

    def __init__(self, config):

        self.cur_epoch = 0
        self.ref_path = "output/ref_pcs/%s.npy" % config.category
        self.syn_path = config.output_dir + "/des_syn_%d.npy" 
        self.save_path = "%s/evaluate.json" % config.output_dir
        self.batch_size = str(config.batch_size)
        self.curr_proc = None

    def do_evaluation(self, epoch):

        cmd = ['python', 'utils/util_torch.py', self.ref_path, self.syn_path, 
                self.batch_size, str(epoch), self.save_path, '1']
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def load_result(self, full=False):

        if not os.path.isfile(self.save_path):
            return {}
        with open(self.save_path, 'r') as f:
            data = f.read() 
            dic = eval("{" + data.replace("\n", "")[:-1] + "}")
        if full: 
            return dic 
        out = {}
        while self.cur_epoch in dic: 
            out[self.cur_epoch] = dic[self.cur_epoch]
            self.cur_epoch += 1
        return out
    
    def last_wait(self):

        while self.curr_proc.poll() is None:
            dic = self.load_result()
            if dic: 
                yield dic 
            time.sleep(5)
        dic = self.load_result()
        if dic: 
            yield dic 

    def add_evaluation(self, epoch):
        
        # Do both add and ask. 
        if self.curr_proc is None or self.curr_proc.poll() is not None: 
            self.curr_proc = self.do_evaluation(epoch)
        return self.load_result() 
       
def save_evaluation_ref(dataloader, category):

    path = "output/ref_pcs/%s.npy" % category
    if not os.path.isfile(path):
        ref_pcs = [pcs for pcs in dataloader]
        ref_pcs = torch.cat(ref_pcs).cpu().data.numpy()
        np.save(path, ref_pcs)
        print(ref_pcs.shape)
        print("Reference pcs saved.")

def exec_eval(config): 

    if config.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda) 
    epoch, cont = config.start_epoch, config.continue_epoch
    run_cnt = 0
    ref_pcs = np.load(config.ref_path)
    print("start")
    while epoch >= -1 and run_cnt < config.continue_epoch: 
        epoch -= 1
        syn_path_real = config.syn_path if epoch == -2 else config.syn_path % epoch
        if not os.path.isfile(syn_path_real):
            continue 
        sample_pcs = np.load(syn_path_real) 
        if type(sample_pcs) is not np.ndarray:
            sample_pcs = sample_pcs['arr_0']
        # print("Starting %s." % syn_path_real)
        res = quantitative_analysis(sample_pcs, ref_pcs, config.batch_size, config.max_num)
        for k in res:
            if type(res[k]) is torch.Tensor:
                res[k] = float(res[k].cpu().data)
        text = "%50s|%4d|%.4f|%.4f|%.4f|%.4f|%.4f|" % (config.syn_path[-50:], epoch, 
                res['jsd']*10, res['lgan_mmd-CD']*100, res['lgan_mmd-EMD']*10, res['lgan_cov-CD'], res['lgan_cov-EMD'])
        if config.save_path is not None:
            with open(config.save_path, 'a+') as outfile:
                outfile.writelines(str(epoch) + " : " + str(res) + ",")
            with open(config.save_path+"s", 'a+') as outfile:
                outfile.writelines(text + "\n")
        print(text)
        # PushNotification().post_text("[GPointNet] Eval res: %s!" % text)
        run_cnt += 1

def quantitative_analysis(sample_pcs, ref_pcs, batch_size=256, num=100000, full=True):
    # Return 6 quantitative results:
    # jsd :
    # mmd_EMD :
    # mmd_CD :
    # cov_EMD :
    # cov_CD :
    # Inception Score by PointNet++

    sample_pcs, ref_pcs = sample_pcs[:num], ref_pcs[:num]
    if ref_pcs.shape[1] < 10: 
        ref_pcs = np.swapaxes(ref_pcs, 1, 2)
    if sample_pcs.shape[1] < 10: 
        sample_pcs = np.swapaxes(sample_pcs, 1, 2)
    if sample_pcs.shape[1] < 2048: 
        ref_pcs = ref_pcs[:, :sample_pcs.shape[1]]
    sample_pcs, ref_pcs = sample_pcs[:, :2048].astype(np.float32), ref_pcs[:, :2048].astype(np.float32)

    if sample_pcs.shape[2] == 2: 
        sample_pcs = np.concatenate([sample_pcs, np.zeros(([sample_pcs.shape[0], sample_pcs.shape[1], 1]))], 2)
        ref_pcs = np.concatenate([ref_pcs, np.zeros(([ref_pcs.shape[0], ref_pcs.shape[1], 1]))], 2)
        
    if ref_pcs.shape[0] > sample_pcs.shape[0]: 
        ref_pcs = ref_pcs[:sample_pcs.shape[0]]
    if sample_pcs.shape[0] > ref_pcs.shape[0]: 
        sample_pcs = sample_pcs[:ref_pcs.shape[0]]
    

    sample_var, ref_var = sample_pcs.std(axis=(0,1)), ref_pcs.std(axis=(0,1))
    # print(sample_pcs.shape, ref_pcs.shape, sample_var, ref_var)
    sample_pcs = sample_pcs / sample_var * ref_var
    # sample_max, sample_min = sample_pcs.max(axis=(0,1), keepdims=True), sample_pcs.min(axis=(0,1), keepdims=True)
    # sample_pcs = ((sample_pcs - sample_min) / (sample_max - sample_min))*2-1
    # ref_max, ref_min = ref_pcs.max(axis=(0,1), keepdims=True), ref_pcs.min(axis=(0,1), keepdims=True)
    # ref_pcs = ((ref_pcs - ref_min) / (ref_max - ref_min))*2-1
    
    sample_pcs_t = torch.from_numpy(sample_pcs).cuda().contiguous()
    ref_pcs_t = torch.from_numpy(ref_pcs).cuda().contiguous()
    res = compute_all_metrics(sample_pcs_t, ref_pcs_t, batch_size, accelerated_cd=True)
    # knn_calculate(sample_pcs, ref_pcs, batch_size)
    # res = EMD_CD(sample_pcs_t, ref_pcs_t, batch_size, accelerated_cd=False)

    # M_rs_cd, M_rs_emd = pairwise_EMD_CD_(sample_pcs_t, ref_pcs_t, batch_size, accelerated_cd=False)
    # res_cd = lgan_mmd_cov(M_rs_cd.t())
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # res['cov_EMD'], res['cov_CD'] = res_emd['lgan_cov'], res_cd['lgan_cov']
    res['jsd'] = jsd_between_point_cloud_sets(sample_pcs, ref_pcs)
    # res['inception'], res['average_prob'] = 0, 0#inception_score(sample_pcs, truth_label=3)

    if full:
        return res
    compact = {
        'jsd': res['jsd'] * 10, 'mmd-CD': res['lgan_mmd-CD']*100, 'mmd-EMD': res['lgan_mmd-EMD']*10, 
        'cov-CD': res['lgan_cov-CD'] * 100, 'cov-EMD': res['lgan_cov-EMD'] * 100
    }
    return compact

class PushNotification(object):

    def __init__(self, general_message=""):

        try: 
            with open(os.path.expanduser(os.path.join("~", ".ssh", "we_key")), "r") as f: 
                self.key = json.load(f)
        except: 
            print("key not existed. Skipped.")
            return None
        self.general_message = general_message

    @classmethod
    def push_error(cls, func, text=""):

        if text == "":
            text = func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            notifier = cls()
            try: 
                res = func(*args, **kwargs)
                notifier.post_text("%s finished!" % text)
                return res
            except Exception as e:
                print(traceback.format_exc()) 
                notifier.post_text("%s FAILED! Error message: %s" % (text, str(traceback.format_exc())))
        return wrapper

    def _get_token(self, wechat_id, secret_key):
        token_url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=%s&corpsecret=%s" % (wechat_id, secret_key)
        res = requests.get(url = token_url).json()
        try:
            return(res["access_token"])
        except:
            return -1

    def _send_message(self, token, content):

        msgurl = "https://qyapi.weixin.qq.com/cgi-bin/appchat/send?access_token=%s" % token
        message = {"chatid" : "experiment", 
              "msgtype": "text",
              "text" : {"content" : content}, 
              "safe" : 0}
        return requests.post(msgurl, data=json.dumps(message))

    def _post_text(self, token, content):

        msgurl = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s" % token
        message = {"touser" : self.key["user"], 
              "msgtype": "text",
              "agentid" : self.key["agent_id"], 
              "text" : {"content" : self.general_message + content}, 
              "safe" : 0}
        return requests.post(msgurl, data=json.dumps(message))
    
    def _post_image(self, token, path):

        msgurl = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s" % token
        media_url = "https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token=%s&type=image" % token
        files = {"media" : open(path, 'rb')}
        res = requests.post(media_url, files=files)
        print(res)
        media_id = res.json()["media_id"]
        message = {"touser" : self.key["user"], 
              "msgtype": "image",
              "agentid" : self.key["agent_id"], 
              "image" : {"media_id" : media_id}, 
              "safe" : 0}
        return requests.post(msgurl, data=json.dumps(message))
    
    def post_text(self, content):
        try: 
            return self._post_text(self._get_token(self.key["wechat_id"], self.key["secret_key"]), content)
        except: 
            print("push failed.")

    def post_image(self, content):
        try: 
            return self._post_image(self._get_token(self.key["wechat_id"], self.key["secret_key"]), content)
        except: 
            print("push failed.")


class PointCloudDataCollator(object):

    def __init__(self, config):

        self.argment_noise = config.argment_noise
        self.num_point = config.num_point
        self.random_sample = config.random_sample
        self.swap_axis = config.swap_axis
        self.normalize = config.normalize

    def __call__(self, pcds):

        out_pcd = []
        for pcd in pcds: 
            if self.random_sample:
                if len(pcd) >= self.num_point: 
                    idx = np.random.permutation(len(pcd))[:self.num_point]
                else:
                    idx = np.concatenate([np.arange(len(pcd)), 
                    np.random.choice(np.arange(len(pcd)), size=self.num_point - len(pcd), replace=True)])
                out_pcd.append(pcd[idx])
            else:
                out_pcd.append(pcd[:, :self.num_point])
        out_pcd = torch.stack(out_pcd)
        out_pcd = out_pcd + torch.randn(out_pcd.shape, device=out_pcd.device) * self.argment_noise
        if self.normalize == "per_shape": 
            mean = torch.mean(out_pcd, dim=1, keepdim=True)
            mean = mean.repeat((1,out_pcd.shape[1],1))
            out_pcd = out_pcd - mean
            mmx, _ = torch.max(torch.abs(out_pcd), dim=1, keepdim=True)
            mmx = mmx.repeat((1,out_pcd.shape[1],1))
            out_pcd = out_pcd / mmx 

        return np.swapaxes(out_pcd, 1, 2) if self.swap_axis else out_pcd


class PointCloudDataSet(torch.utils.data.dataset.Dataset): 

    def __init__(self, config):

        category = config.category 

        cate_temp = category.split("_")[0]
        train_data = []
        if cate_temp == "modelnet40":
            categories = ['cup', 'bookshelf', 'lamp', 'stool', 'desk', 'toilet', 'night_stand', 'bowl', 'door', 'flower_pot', 'plant', 'stairs', 'bottle', 'mantel', 'sofa', 'laptop', 'xbox', 'tent', 'piano', 'car', 'wardrobe', 'tv_stand', 'cone', 'range_hood', 'bathtub', 'curtain', 'sink', 'glass_box', 'bed', 'chair', 'person', 'radio', 'dresser', 'bench', 'airplane', 'guitar', 'keyboard', 'table', 'monitor', 'vase']
            for cat in categories:
                d = np.load('data/%s_train.npy' % cat)
                train_data.append(d)

        elif cate_temp == "modelnet10":
            categories = ['desk', 'toilet', 'night_stand', 'sofa', 'bathtub', 'bed', 'chair', 'dresser', 'table', 'monitor']
            for cat in categories:
                d = np.load('data/%s_train.npy' % cat)
                train_data.append(d)

        elif cate_temp == "partnet": 

            partnet_base = "/home/fei960922/Documents/Dataset/PointCloud/partnet/sem_seg_h5"
            if len(category.split("_")) > 1:
                categories = [category.split("_")[1]]
            else: 
                categories = os.listdir(partnet_base)
            for cat in categories:
                for name in [s for s in os.listdir("%s/%s" % (partnet_base, cat)) if s[:3]=="tra" and s[-1]=="5"]:
                    h5file = h5py.File("%s/%s/%s" % (partnet_base, cat, name))
                    d = np.asarray(h5file['data'])
                    # seg = h5file['label_seg']
                    train_data.append(d)

        elif cate_temp == "shapenetpart":
            pcd, label, seg = [], [], []
            for i in range(10):
                try:
                    f = h5py.File("data/hdf5_data/ply_data_train%d.h5" % i)
                    pcd.append(f['data'][:])
                    label.append(f['label'][:])
                    seg.append(f['pid'][:])
                except:
                    break 
            train_data, train_label, train_seg = np.concatenate(pcd), np.concatenate(label), np.concatenate(seg)

            # data stored in list as a pickle file. range from [-1, 1]
            # path = "data/shapenetpart_training.pkl" 
            # with open(path, "rb") as f: 
            #     data = pickle.load(f)
            if len(category.split("_")) > 1:
                idx = int(category.split("_")[1])
                print((train_label == idx).shape)
                train_data = train_data[(train_label == idx).squeeze()]

        elif cate_temp == "mnist":
            
            train_data = pickle.load(open("data/mnist_normal.pkl", "rb"))
            # for pnt in temp_data:
            #     train_data.append(np.concatenate([pnt, np.zeros(([pnt.shape[0], 1]))], 1))
        else: 
            train_data = [np.load('data/%s_train.npy' % category)]
        

        if len(train_data[0].shape) == 3:
            train_data = np.concatenate(train_data)

            # Normalize to [-1, 1]
            if config.normalize == 'ebp':
                max_val = train_data.max()
                min_val = train_data.min()
                train_data = ((train_data - min_val) / (max_val - min_val)) * 2 - 1
            elif config.normalize == 'scale': 
                train_data = train_data - train_data.mean(axis=(0,1))
                max_val = np.abs(train_data).max(axis=(0,1))
                train_data = train_data / max_val 

            if config.argment_mode:
                print("Argment data by rotating through x axis. %d Data" % len(train_data))
                train_data = self._data_argment(train_data)
                print("Argmented. %d Data" % len(train_data))
        else: 
            # data stored in pickle
            if config.argment_mode:
                print("Argment data by rotating through x axis. %d Data" % len(train_data))
                new_train_data = []
                for pcd in train_data:
                    for xrot in (np.arange(-1, 1, 0.25) * math.pi):
                        M = euler2mat(0, xrot, 0)
                        new_train_data.append(np.dot(pcd, M.transpose()))
                train_data = new_train_data
                print("Argmented. %d Data" % len(train_data))

        config.data_size = min(config.data_size, len(train_data))
        idx = np.random.permutation(len(train_data))[:config.data_size]
        if type(train_data) is list: 
            train_data = [train_data[i] for i in idx]
        else:
            train_data = train_data[idx]
        self.data = train_data

    def _data_argment(self, train_data): 

        # Enriched Normailzation 

        # Rotate through X-axis 
        out_data = []
        for xrot in (np.arange(-1, 1, 0.25) * math.pi):
            M = euler2mat(0, xrot, 0)
            out_data.append(np.dot(train_data, M.transpose()))
        return np.concatenate(out_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float)


# ----------------------------------------
# Inherit from PointNet
# Author: Charles R. Qi, Hao Su
# ----------------------------------------


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        px[px < 0] = 0
        px[px > image.shape[0]-1] = 0
        py[py < 0] = 0
        py[py > image.shape[1]-1] = 0
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    image[0,0] = 0
    image = image / np.max(image)
    return image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ref_path', type=str, default="default")
    parser.add_argument('-syn_path', type=str, default="default")
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-start_epoch', type=int, default=-1)
    parser.add_argument('-save_path', type=str, default=None)
    parser.add_argument('-max_num', type=int, default=100000)
    parser.add_argument('-continue_epoch', type=float, default=1)
    parser.add_argument('-cuda', type=int, default=-1)
    parser.add_argument('-verbose', type=int, default=1)
    config = parser.parse_args()
    exec_eval(config)