# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import sys
sys.path.append(".")
from utils.util_torch import visualize
try:
    sys.path.append("/home/fei960922/Programs/mitsuba2/build/dist/python")
    import mitsuba
except:
    pass
    # print("mitsuba load failed. It is required if mode=4 is used.")

import numpy as np

def print_mitsuba(pcl, output_filename):

    def standardize_bbox(pcl, points_per_object):
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt_indices)
        pcl = pcl[pt_indices] # n by 3
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = ( mins + maxs ) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
        return result*1.4

    xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,2,2" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            
            <sampler type="independent">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="200"/>
                <integer name="height" value="200"/>
                <rfilter type="gaussian"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
        
    """

    xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

    def colormap(x,y,z):
        vec = np.array([x,y,z])
        vec = np.clip(vec, 0.001,1.0)
        norm = np.sqrt(np.sum(vec**2))
        vec /= norm
        return [136/255, 170/255, 220/255]
        return [vec[0], vec[1], vec[2]]
    xml_segments = [xml_head]

    pcl = standardize_bbox(pcl, pcl.shape[0])
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    pcl[:,2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open('output/mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)

    mitsuba.set_variant('scalar_rgb')
    # mitsuba.set_variant('gpu_autodiff_rgb')
    mitsuba.core.Thread.thread().file_resolver().append(".")
    scene = mitsuba.core.xml.load_file('output/mitsuba_scene.xml')
    scene.integrator().render(scene, scene.sensors()[0])

    # After rendering, the rendered data is stored in the film
    film = scene.sensors()[0].film()
    film.set_destination_file('output/temp.exr')
    film.develop()

    # Write out a tonemapped JPG of the same rendering
    bmp = film.bitmap(raw=True)
    bmp.convert(mitsuba.core.Bitmap.PixelFormat.RGB, mitsuba.core.Struct.Type.UInt8, srgb_gamma=True).write(output_filename)
    bmp_linear_rgb = bmp.convert(mitsuba.core.Bitmap.PixelFormat.RGB, mitsuba.core.Struct.Type.Float32, srgb_gamma=False)
    image_np = np.array(bmp_linear_rgb)
    print(image_np.shape)

def reprint_202007():
    num_sample = 10000
    num_pts = 1024
    category_list = ["night_stand", "monitor", "dresser", "desk", "bathtub", "table", "sofa", "bed", "toilet", "chair"]
    # category_list = ["bathtub", "table", "sofa", "bed", "toilet"]

    # d = np.load("output/war_start_smaller/synthesis/des_syn_980.npy")
    # visualize(d[:num_sample, :num_pts], output_filename="dropbox/Synthesis/chair_warm", mode=3)
            
    # GroundTruth
    # print("Reprint Ground Truth")
    # for cate in category_list:
    #     ref_pcs = np.load("data/%s_train.npy" % cate)
    #     visualize(ref_pcs[:num_sample, :num_pts], output_filename="dropbox/GroundTruth/%s" % cate, mode=3)

    # # Synthesis Results 

    output_path = {
        # 'ebm' : "./output/fixed_noise/full_%s/synthesis/des_syn_%d.npy", 
        'ebm_random_2048' : "./output/pre_ECCV/random_2048/%s/des_syn_%d.npy", 
        'flow' : "../PointCloud_reference/PointFlow/checkpoints/jerry/%s/syn_flow_%d.npy", 
        'lgan' : "../PointCloud_reference/latent_3d_points/output_1024/%s/lgan_out/latent_gan_with_chamfer_ae/epoch_%d.npz", 
        'rgan' : "../PointCloud_reference/latent_3d_points/output_1024/%s/rgan_out/raw_gan_with_w_gan_loss/syn_rgan_epoch%d.npz"
    }

    for class_name in category_list:
        names = {'ebm_random_2048'}
        # names = {'lgan', 'rgan'} #, 'gmm', 'i-gan', 'flow'
        for name in names: 
            
            data = None
            for k in [1000, 950, 250, 249, 100, 99, 60, 59, 40, 39, 30, 29, 10, 1]:
                try:
                    data = np.load(output_path[name] % (class_name, k))
                    break
                except Exception as e:
                    pass
            syn_term = data if type(data) is np.ndarray else data['arr_0']
            max_val = syn_term.max()
            min_val = syn_term.min()
            syn_term = ((syn_term - min_val) / (max_val - min_val)) * 2 - 1
            data = syn_term[:] 
            print(class_name, name, data.shape, np.var(data), np.mean(data))
            for i in range(10, data.shape[0]):
                print_mitsuba(data[i, :1024], output_filename="output/dropbox/test/%s_%s_%d.png" % (class_name, name, i))
            # visualize(data[:num_sample], output_filename="dropbox/Synthesis/%s_%s" % (class_name, name), mode=3)
            # visualize(data[:10], 1, 1,output_filename="dropbox/OLD_MODE_syn_%s_%s.png" % (class_name, name), mode=1)



    # # Reconstruction Results 

#     output_path = {
#         'ebm_new' : "./output/rec_raw/reconstruction_ebm_%s.npy", 
#         'flow' : "../../PointCloud_reference/PointFlow/checkpoints/reconstruction/reconstruction_[\'%s\']_flow.npy", 
#     }

#     for class_name in category_list:

#         names = {'flow'} #, 'gmm', 'i-gan', 'flow'
#         for name in names:
#             data = np.load(output_path[name] % class_name)
#             syn_term = data if type(data) is np.ndarray else data['arr_0']
#             max_val = syn_term.max()
#             min_val = syn_term.min()
#             syn_term = ((syn_term - min_val) / (max_val - min_val)) * 2 - 1
#             data = syn_term[:, :1024]
#             print(class_name, name, data.shape, np.var(data), np.mean(data))
#             visualize(data[:num_sample], output_filename="dropbox/Reconstruction/%s_%s" % (class_name, name), mode=3)

# def output_tex():

#     template_g = "\\includegraphics[trim=60 60 60 60,clip,width=0.105\\linewidth]{./figures/results/GroundTruth/%s/%d.png}"
#     template_e = "\\includegraphics[trim=60 60 60 60,clip,width=0.105\\linewidth]{./figures/results/Reconstruction/%s_ebm_new/%d.png}"
#     template_f = "\\includegraphics[trim=60 60 60 60,clip,width=0.105\\linewidth]{./figures/results/Reconstruction/%s_flow/%d.png}"

#     chair_idx = [3,5,7,8,19,24,36,41,46,56,68]
#     toilet_idx = [21,29,44,45,62,65,72,91,143,274]
#     table_idx = [115,117,120,180,199,239]

#     for idx in chair_idx:
#         print(template_g % ('chair', idx))
#         print(template_e % ('chair', idx))
#         print(template_f % ('chair', idx))
#     for idx in toilet_idx:
#         print(template_g % ('toilet', idx))
#         print(template_e % ('toilet', idx))
#         print(template_f % ('toilet', idx))
#     for idx in table_idx:
#         print(template_g % ('table', idx))
#         print(template_e % ('table', idx))
#         print(template_f % ('table', idx))

def reprint_camera_ready():

    num_sample = 100
    num_pts = 2048
    category_list = ["night_stand", "monitor", "dresser", "desk", "bathtub", "table", "sofa", "bed", "toilet", "chair"]
    # category_list = ["bathtub", "table", "sofa", "bed", "toilet"]

    # d = np.load("output/war_start_smaller/synthesis/des_syn_980.npy")
    # visualize(d[:num_sample, :num_pts], output_filename="dropbox/Synthesis/chair_warm", mode=3)
            
    # GroundTruth
    # print("Reprint Ground Truth")
    # for cate in category_list:
    #     ref_pcs = np.load("data/%s_train.npy" % cate)
    #     visualize(ref_pcs[:num_sample, :num_pts], output_filename="dropbox/GroundTruth/%s" % cate, mode=3)

    # # Synthesis Results 

    output_path = {
        'ebm' : "./output/a_bridge/%s_default_0323bm/des_syn_%d.npy", 
        'ebm_old' : "./output/pre_ECCV/random_2048/%s/des_syn_%d.npy", 
        'flow' : "../PointCloud_reference/PointFlow/checkpoints/modelnet/%s/flow_syn_%d.npy", 
        'lgan' : "../PointCloud_reference/latent_3d_points/output_1024/%s/lgan_out/latent_gan_with_chamfer_ae/epoch_%d.npz", 
        'rgan' : "../PointCloud_reference/latent_3d_points/output_1024/%s/rgan_out/raw_gan_with_w_gan_loss/syn_rgan_epoch%d.npz"
    }

    for class_name in category_list:
        names = {'ebm', 'ebm_old', 'flow'}
        # names = {'lgan', 'rgan'} #, 'gmm', 'i-gan', 'flow'
        for name in names: 
            
            data = None
            for k in np.arange(4000, 0, -1):
                if os.path.isfile(output_path[name] % (class_name, k)):
                    data = np.load(output_path[name] % (class_name, k))
                    print("%s found." % (output_path[name] % (class_name, k)))
                    break
            if data is None: 
                print("%s_%s not found, skipped." % (name, class_name))
                continue 

            syn_term = data if type(data) is np.ndarray else data['arr_0']
            print(class_name, name, data.shape, np.var(data), np.mean(data))
            if syn_term.shape[1] < 10: 
                syn_term = np.swapaxes(syn_term, 1, 2)
            # for i in range(10, data.shape[0]):
            #     print_mitsuba(data[i, :1024], output_filename="output/dropbox/test/%s_%s_%d.png" % (class_name, name, i))
            visualize(syn_term[:num_sample], output_filename="output/dropbox/Synthesis/%s_%s" % (class_name, name), mode=3)
            # visualize(data[:10], 1, 1,output_filename="dropbox/OLD_MODE_syn_%s_%s.png" % (class_name, name), mode=1)


def reprint_rec(max_num=128):

    out_path = "output/dropbox/syn_mitsuba/reconstruction_black/%s"
    path = {
        "gt": "data/%s_train.npy", 
        "ebp": "output/a_bridge/test/%s_new_rec.npy",
        "flow": "../PointCloud_reference/PointFlow/checkpoints/modelnet/%s/flow_rec.npy",
    }
    for class_name in ["toilet","table", "bathtub", "monitor", "night_stand", "dresser", "bed",  "desk", "sofa"]: 
        data = {}
        for k, v in path.items():
            data[k] = np.load(path[k] % class_name)
            if data[k].shape[1] == 3:
                data[k] = np.swapaxes(data[k], 1, 2)
        if not os.path.exists(out_path % class_name):
            os.mkdir(out_path % class_name)
        for i in range(min(max_num, data["ebp"].shape[0])): 
            for k, v in path.items():
                visualize(data[k][i:i+1], num_cols=1, num_rows=1, output_filename=out_path % class_name + "/%d-%s.png" % (i, k), mode=1)
                # print_mitsuba(data[k][i], out_path % class_name + "%d-%s.png" % (i, k))
            print("%d printed" % i)

def reprint_path(path, max_num, out_path):

    data = np.load(path)
    if data.shape[1] == 3:
        data = np.swapaxes(data, 1, 2)

    max_val = data.max()
    min_val = data.min()
    data = ((data - min_val) / (max_val - min_val)) * 2 - 1
    
    if out_path is None: 
        out_path = path.split("/")[-1][:-4]
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    visualize(data[:max_num], output_filename=out_path, mode=3)
    # out_path = out_path + "/%d.png"
    # for i in range(min(max_num, data.shape[0])): 
    #     print_mitsuba(data[i], out_path % i)
    #     print("%d printed" % i)

if __name__ == "__main__":
    
    # save_to_xyz("output/all_2048/synthesis/des_syn_21.npy", "ebm")
    # save_to_xyz("data/chair_train.npy", "truth")
    if len(sys.argv)>1:
        reprint_path(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100000, sys.argv[3] if len(sys.argv) > 3 else None)
    else: 
        reprint_rec()