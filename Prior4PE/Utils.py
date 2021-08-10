import numpy as np
import cv2

from sklearn.cluster import MeanShift
    
def cluster(seg, d, c_off, config):
    xy_coords = (np.stack(np.meshgrid(np.arange(seg.shape[1]), np.arange(seg.shape[0])), axis=-1) + 0.5) * config['outputStrides']

    res_seg = seg[...,0] > 0.75
    res_coord = xy_coords[res_seg]
    res_d = d[res_seg]
    res_coff = c_off[res_seg]
    
    X = np.concatenate([res_coord + res_coff, res_d], axis=-1)
    clustering = MeanShift(bandwidth=config['MeanShift_bandwidth']).fit(X)
    
    if config['verbose'] > 0:
        print("Number of clusters found:", clustering.labels_.max()+1)
    labels = np.zeros_like(res_seg, int)
    labels[res_seg] = clustering.labels_ + 1
    
    
    means_with_count = []
    for bbi in range(clustering.labels_.max()+1):
        means_with_count.append((X[clustering.labels_ == bbi].mean(axis=0), (clustering.labels_ == bbi).sum()))            
    return means_with_count, labels

def coord_K_from(mean_bb, config):
    scale = mean_bb[2:].max() / config['pei_xyDim']
    new_bbs = mean_bb[:2] - mean_bb[2:]/2. + (mean_bb[2:] - mean_bb[2:].max()) / 2
    
    return np.stack([np.array([scale,scale]), new_bbs])

def tranform(input_img, coord_K, config):
    warp_mat = cv2.invertAffineTransform(np.array([
        [coord_K[0,0], 0,  coord_K[1,0]],
        [0, coord_K[0,1],  coord_K[1,1]]
    ]))
    warp_dst = cv2.warpAffine(input_img, warp_mat, (config['pei_xyDim'],config['pei_xyDim']))
    return warp_dst

def tranformRGBDIseg(coord_K, rgb, d, iseg, config):
    _rgb = tranform(rgb, coord_K, config)
    _d = tranform(d, coord_K, config) if not d is None else np.zeros_like(_rgb[...,0])
    _iseg = tranform(iseg, coord_K/config['outputStrides'], config)[...,np.newaxis]
    return _rgb, _d, _iseg
    