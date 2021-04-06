import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm, Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
# try:
#     from torch_geometric.datasets import ModelNet
#     import torch_geometric.transforms as T
#     from torch_geometric.data import DataLoader
#     from torch_geometric.nn import PointConv, fps, radius, global_max_pool
# except:
#     pass

class energy_point_default(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        net_local, net_global = [], [] 
        prev = config.point_dim
        self.pnt_axis = 2 if config.swap_axis else 1
        for h in config.hidden_size[0]: 
            layer = residual_Conv1d(h) if h==prev else torch.nn.Conv1d(prev, h, 1) 
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02) 
            # torch.nn.init.zeros_(layer.bias) 
            net_local.append(layer)
            if config.batch_norm == "bn": 
                net_local.append(torch.nn.BatchNorm1d(h)) # Question exists
            elif config.batch_norm == "ln": 
                net_local.append(torch.nn.LayerNorm(config.num_point))
            elif config.batch_norm == "lnm": 
                net_local.append(torch.nn.LayerNorm([h, config.num_point]))
            elif config.batch_norm == "in": 
                net_local.append(torch.nn.InstanceNorm1d(h)) # Question exists
            if config.activation != "":
                net_local.append(getattr(torch.nn, config.activation)())
            prev = h
        for h in config.hidden_size[1]: 
            layer = residual_Linear(h) if h==prev else torch.nn.Linear(prev, h) 
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02) 
            # torch.nn.init.zeros_(layer.bias) 
            net_global.append(layer)
            if config.activation != "":
                net_global.append(getattr(torch.nn, config.activation)())
            prev = h
        net_global.append(torch.nn.Linear(prev, 1))
        self.local = torch.nn.Sequential(*net_local)
        self.globals = torch.nn.Sequential(*net_global)
    
    def forward(self, point_cloud, out_local=False, out_every_layer=False):

        local = self.local(point_cloud)
        if out_local:
            return local
        out = self.globals(torch.mean(local, self.pnt_axis))
        return out 

    def _output_all(self, pcs):
        
        res = [] 
        for layer in self.local: 
            pcs = layer(pcs) 
            if type(layer) is torch.nn.LayerNorm: 
                res.append(pcs)
        return res 

class residual_Conv1d(torch.nn.Module):

    def __init__(self, h):
        super().__init__()
        self.layer = torch.nn.Conv1d(h, h, 1) 
    def forward(self, x):
        return self.layer(x) + x

class residual_Linear(torch.nn.Module):

    def __init__(self, h):
        super().__init__()
        self.layer = torch.nn.Linear(h, h) 
    def forward(self, x):
        return self.layer(x) + x

class energy_point_residual(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        net_local, net_global = [], [] 
        prev = config.point_dim
        self.pnt_axis = 2 if config.swap_axis else 1
        for h in config.hidden_size[0]: 
            layer = torch.nn.Conv1d(prev, h, 1) 
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02) 
            # torch.nn.init.zeros_(layer.bias) 
            net_local.append(layer)
            if config.batch_norm == "bn": 
                net_local.append(torch.nn.BatchNorm1d(h)) # Question exists
            elif config.batch_norm == "ln": 
                net_local.append(torch.nn.LayerNorm(config.num_point))
            elif config.batch_norm == "lnm": 
                net_local.append(torch.nn.LayerNorm([h, config.num_point]))
            elif config.batch_norm == "in": 
                net_local.append(torch.nn.InstanceNorm1d(h)) # Question exists
            if config.activation != "":
                net_local.append(getattr(torch.nn, config.activation)())
            prev = h
        for h in config.hidden_size[1]: 
            layer = torch.nn.Linear(prev, h) 
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02) 
            # torch.nn.init.zeros_(layer.bias) 
            net_global.append(layer)
            if config.activation != "":
                net_global.append(getattr(torch.nn, config.activation)())
            prev = h
        net_global.append(torch.nn.Linear(prev, 1))
        self.local = torch.nn.Sequential(*net_local)
        self.globals = torch.nn.Sequential(*net_global)
    
    def forward(self, point_cloud, out_local=False):

        local = self.local(point_cloud)
        if out_local:
            return torch.mean(local, self.pnt_axis) 
        out = self.globals(torch.mean(local, self.pnt_axis))
        return out 

class energy_point_pointnet(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(energy_point_pointnet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# Inherit from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class energy_point_pointnet2(torch.nn.Module):
    def __init__(self, config=None):
        super(energy_point_pointnet2, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)
        self.batch_i = torch.arange(config.batch_size, device=config.device).repeat(config.num_point,1).t().reshape(-1)

    def forward(self, data):
        
        sa1_out = self.sa1_module(None, data.view(-1, 3), self.batch_i[:data.shape[0]*data.shape[1]])
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x
