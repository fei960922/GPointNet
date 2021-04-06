from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import h5py
import requests
import json
import functools
import traceback
import math 
import pickle 

from tqdm import tqdm
from utils.eulerangles import euler2mat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

BASE_URL = 'https://drive.google.com/a/g.ucla.edu/file/d/17GapKvQDMTNJWM4L-_KZgicahA08-TZE/view?usp=sharing'

def download(data_path='data'):
    # Download dataset for point cloud classification
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    download_modelnet40(data_path)

def download_modelnet40(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    data_dir = 'modelnet40_2048_category_hdf5'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found ModelNet40 - skip')
        return
    filename, drive_id = "modelnet40_2048_category_hdf5.zip", "17GapKvQDMTNJWM4L-_KZgicahA08-TZE"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)
    zipfile = 'modelnet40_2048_category_hdf5.zip'
    os.system('unzip %s' % save_path)
    os.system('mv %s %s' % (zipfile[:-4], dirpath))
    os.system('rm %s' % save_path)

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def rotate_point_cloud(batch_data, xrot):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename, dataset='data', dtype=np.float32):
    f = h5py.File(h5_filename, 'r')
    data = np.asarray(f[dataset], dtype=dtype)
    return data

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def save_h5_to_npy(target_dir = '../data/modelnet40_2048_category_hdf5'):
    
    categorys = os.listdir(target_dir)
    print(categorys)
    # categorys = ["chair", "airplane", "bed", "toilet"]
    # categorys = ["toilet", "table", "sofa", "night_stand", "monitor", "dresser", "desk", "chair", "bed", "bathtub"]
    for cat in categorys:
        data = loadDataFile(os.path.join(target_dir, cat, cat + '_train.h5'))
        print(cat, data.shape)
        np.save("../data/" + cat + "_train.npy", data)
        data = loadDataFile(os.path.join(target_dir, cat, cat + '_test.h5'))
        print(cat, data.shape)
        np.save("../data/" + cat + "_test.npy", data)

# Written by Jerry Xu 
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
        
def data_argment(train_data): 

    # Enriched Normailzation 

    # Rotate through X-axis 
    out_data = []
    for xrot in (np.arange(-1, 1, 0.25) * math.pi):
        M = euler2mat(0, xrot, 0)
        out_data.append(np.dot(train_data, M.transpose()))
    return np.concatenate(out_data)

def load_data(category, arg_mode):

    cate_temp = category.split("_")[0]
    train_data = []
    if cate_temp == "modelnet40":
        categories = ['cup', 'bookshelf', 'lamp', 'stool', 'desk', 'toilet', 'night_stand', 'bowl', 'door', 'flower_pot', 'plant', 'stairs', 'bottle', 'mantel', 'sofa', 'laptop', 'xbox', 'tent', 'piano', 'car', 'wardrobe', 'tv_stand', 'cone', 'range_hood', 'bathtub', 'curtain', 'sink', 'glass_box', 'bed', 'chair', 'person', 'radio', 'dresser', 'bench', 'airplane', 'guitar', 'keyboard', 'table', 'monitor', 'vase']
        for cat in categories:
            d = np.load('data/%s_train.npy' % cat)
            print(d.var(), d.mean(), d.max(), d.min())
            train_data.append(d)
    elif cate_temp == "modelnet10":
        for cat in categories:
            d = np.load('data/%s_train.npy' % cat)
            print(d.var(), d.mean(), d.max(), d.min())
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
                print(d.shape, d.var(), d.mean(), d.max(), d.min())
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
            train_data = train_data[train_label == idx]
    elif cate_temp == "mnist":
        # train_data = []
        train_data = pickle.load(open("data/mnist_normal.pkl", "rb"))
        # for pnt in temp_data:
        #     train_data.append(np.concatenate([pnt, np.zeros(([pnt.shape[0], 1]))], 1))
    else: 
        train_data = [np.load('data/%s_train.npy' % category)]
    if len(train_data[0].shape) == 3:
        train_data = np.concatenate(train_data)
        if arg_mode == 1:
            print("Argment data by rotating through x axis. %d Data" % len(train_data))
            train_data = data_argment(train_data)
            print("Argmented. %d Data" % len(train_data))
    else: 
        # data stored in pickle
        if arg_mode == 1:
            print("Argment data by rotating through x axis. %d Data" % len(train_data))
            new_train_data = []
            for pcd in train_data:
                for xrot in (np.arange(-1, 1, 0.25) * math.pi):
                    M = euler2mat(0, xrot, 0)
                    new_train_data.append(np.dot(pcd, M.transpose()))
            train_data = new_train_data
            print("Argmented. %d Data" % len(train_data))

    idx = np.random.permutation(len(train_data))
    if type(train_data) is list: 
        train_data = [train_data[i] for i in idx]
    else:
        train_data = train_data[idx]
    return train_data

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        # Updated by Jerry
        max_val = self.all_points.max()
        min_val = self.all_points.min()
        self.all_points = ((self.all_points - min_val) / (max_val - min_val)) * 2 - 1

        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]
        return np.swapaxes(tr_out, 0, 1)
        # return {
        #     'idx': idx,
        #     'train_points': tr_out,
        #     'test_points': te_out,
        #     'mean': m, 'std': s, 'cate_idx': cate_idx,
        #     'sid': sid, 'mid': mid
        # }


class ModelNet40PointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ModelNet40.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.sample_size = tr_sample_size
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 40, "%s %s" % (len(self.cates), self.cates)

        # For non-aligned MN
        # self.gravity_axis = 0
        # self.display_axis_order = [0,1,2]

        # Aligned MN has same axis-order as SN
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ModelNet40PointClouds, self).__init__(
            root_dir, self.cates, tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size, split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


class ModelNet10PointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ModelNet10.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 10

        # That's prealigned MN
        # self.gravity_axis = 0
        # self.display_axis_order = [0,1,2]

        # Aligned MN has same axis-order as SN
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ModelNet10PointClouds, self).__init__(
            root_dir, self.cates, tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size, split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def _get_MN40_datasets_(args, data_dir=None):
    tr_dataset = ModelNet40PointClouds(
        split='train',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True)
    te_dataset = ModelNet40PointClouds(
        split='test',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )

    return tr_dataset, te_dataset


def _get_MN10_datasets_(args, data_dir=None):
    tr_dataset = ModelNet10PointClouds(
        split='train',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True)
    te_dataset = ModelNet10PointClouds(
        split='test',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


def get_datasets(args):
    if args.dataset_type == 'shapenet15k':
        tr_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='train',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            random_subsample=True)
        te_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='val',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            all_points_mean=tr_dataset.all_points_mean,
            all_points_std=tr_dataset.all_points_std,
        )
    elif args.dataset_type == 'modelnet40_15k':
        tr_dataset, te_dataset = _get_MN40_datasets_(args)
    elif args.dataset_type == 'modelnet10_15k':
        tr_dataset, te_dataset = _get_MN10_datasets_(args)
    else:
        raise Exception("Invalid dataset type:%s" % args.dataset_type)

    return tr_dataset, te_dataset


def get_clf_datasets(args):
    return {
        'MN40': _get_MN40_datasets_(args, data_dir=args.mn40_data_dir),
        'MN10': _get_MN10_datasets_(args, data_dir=args.mn10_data_dir),
    }


def get_data_loaders(args):
    tr_dataset, te_dataset = get_datasets(args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    train_unshuffle_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
        'train_unshuffle_loader': train_unshuffle_loader,
    }
    return loaders


if __name__ == "__main__":
    shape_ds = ShapeNet15kPointClouds(categories=['airplane'], split='val')
    x_tr, x_te = next(iter(shape_ds))
    print(x_tr.shape)
    print(x_te.shape)



if __name__ == '__main__':
    # download_modelnet40("../data/modelnet40")
    save_h5_to_npy()
    base_dir = 'data/modelnet40_ply_hdf5_2048'
    target_dir = 'data/modelnet40_2048_category'
    train_files = getDataFiles(os.path.join(base_dir, 'test_files.txt'))
    # print(train_files)
    # TEST_FILES = getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

    shape_names = []
    with open(os.path.join(base_dir, 'shape_names.txt'), 'r') as f:
        shape_names = [line.replace('\n', '') for  line in f.readlines()]
    print(shape_names)

    data = []
    label = []
    for fn in range(len(train_files)):
        print('----' + str(fn) + '-----')
        current_data, current_label = loadDataFile(train_files[fn])
        data.append(current_data)
        label.append(current_label)
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)
    print(data.shape)
    print(label.shape)

    phase = 'test'

    for i, shape in enumerate(shape_names):
        indices = np.asarray([ind for ind, l in enumerate(label) if l == i])
        shape_data = data[indices]
        dest_dir = os.path.join(target_dir, shape)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with h5py.File(os.path.join(dest_dir, '%s_%s.h5' % (shape, phase)), 'w')  as h5:
            h5.create_dataset('data', data=shape_data)
        print(dest_dir, shape_data.shape)
