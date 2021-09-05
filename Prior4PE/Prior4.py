import tensorflow as tf
import numpy as np
from cv2 import imread, IMREAD_ANYDEPTH, IMREAD_UNCHANGED

from Prior4PE import Utils, Config
from Prior4PE.Config import model_infos

import cv2

class Prior4:
    def __init__(self, _config):
        self.config = Config.initial_config
        self.config.update(_config)

        self.seg_model = tf.keras.models.load_model(self.config["data_path"] + self.config["seg_model_name"], compile=False)
        if self.seg_model is None:
            raise ValueError(f'Segmentation model not found!{self.config["data_path"] + self.config["seg_model_name"]}')
        print(f'Segmentation model ({self.config["seg_model_name"]}) loaded successfully')
        
        self.pose_model = tf.keras.models.load_model(self.config["data_path"] + self.config["pose_model_name"], compile=False)
        if self.pose_model is None:
            raise ValueError(f'Pose estimation model not found!{self.config["data_path"] + self.config["pose_model_name"]}')
        print(f'Pose estimation model ({self.config["pose_model_name"]}) loaded successfully')
        

    def __call__(self, input_rgb_image, input_d_image=None, cm_input=Config.exampleCameraMatrix, priors=[], lookAt_t = None):
    
        if lookAt_t is None:
            x_off = 0. # 370968.1525 / 1235.32 - 320
            y_off = 0. # 181921.145 / 1235.32 - 240
            s= 1. # 1235.32 / 1000 
        else:
            uvw = cm_input[0] @ np.array(lookAt_t)
            print(lookAt_t.shape, uvw.shape, cm_input.shape)
            uv = uvw[:2] / uvw[2]
            s = uvw[2] / 1000
            x_off = uv[0] - cm_input[0,0,2]
            y_off = uv[1] - cm_input[0,1,2]
            
        warp_mat = cv2.invertAffineTransform(np.array([
            [2.* s, 0,  0 + x_off * 2],
            [0, 2. * s,  0 + y_off * 2]
        ]))
        input_rgb_image = cv2.warpAffine(input_rgb_image, warp_mat, (640, 480))
        used_rgb_image = input_rgb_image[np.newaxis]
        
#         res_seg, res_d, res_coff, used_rgb_image = self.seg_model.predict(input_rgb_image[np.newaxis])
        res_seg, res_d, res_coff, _,_,_ = self.seg_model.predict(input_rgb_image[np.newaxis])
        means_with_count, labels = Utils.cluster(res_seg[0], res_d[0], res_coff[0], self.config)

        outputs_list = []
        mean_bb_list = []
        for clusterid in range(len(means_with_count)):
            if self.config["verbose"] > 1:
                print("=====================================")
            if means_with_count[clusterid][1] < 100:
                if self.config["verbose"] > 0:
                    print('Skipped cluster with', means_with_count[clusterid][1], "pixels.")
                continue

            iseg = (labels == clusterid + 1).astype(float)

            mean_bb = means_with_count[clusterid][0]
            coord_K = Utils.coord_K_from(mean_bb, self.config)

            mean_bb_list.append(mean_bb)
            
            if self.config["verbose"] > 1:
                print("coord_K", coord_K)

            _rgb, _d, _iseg = Utils.tranformRGBDIseg(coord_K, used_rgb_image[0], input_d_image, iseg, self.config)

            coord_K = coord_K * s
            coord_K[1] += np.array([x_off, y_off])
            if self.config["verbose"] > 1:
                print("corr coord_K", coord_K)
                
            outputs = self.pose_model.predict(tf.data.Dataset.from_tensors((tuple(
                [_rgb[np.newaxis], _iseg[np.newaxis], cm_input, coord_K[np.newaxis], _d[np.newaxis]]
                + list(priors)),
                ())))
            
            outputs_list.append(outputs)
            if self.config["verbose"] > 1:
                self.print_eval_output(outputs, priors)
                
        return outputs_list, mean_bb_list, labels
    
    def print_eval_output(self, outputs, priors):
        countOfPriors = priors[0].shape[1]
        
        print(outputs[0])
        print()
        print('Found a ', ['Plate', 'Cup'][outputs[0].argmax()])
        print()
        print('Pose for rgbSmall variant:')
        print(outputs[1][0,0])
        print()
        if outputs[0].argmax() < 1:
            print('Pose for rgbBig variant')
            print(outputs[1][0,1])
        print()

        mss = outputs[2][0,4:4+countOfPriors*2] if outputs[0].argmax() < 1 else outputs[2][0,4:4+countOfPriors]
        print('ms prior values', mss)
        print()
        print('prior Pose variant:')
        print(outputs[1][0,4+mss.argmin()])
        if outputs[0].argmax() < 1:
            print(f'The found plate is a {"SMALL" if mss.argmin() < countOfPriors else "BIG"} one!')
        print()

        print(outputs[1][0,2:4])
        print(outputs[2][0,2:4])


        print(outputs[1][0,4+countOfPriors*2:4+countOfPriors*4])
        print(outputs[2][0,4+countOfPriors*2:4+countOfPriors*4])


        print(outputs[2][0])

        print("=====================================")
    
    def test(self, lookAt_t = None):
        config = self.config
        self.config.update({"verbose": 10})
        
        from Prior4PE import PriorDecl
        
        test_rgb_image = imread(self.config["data_path"] + Config.exampleRGBImageName, IMREAD_UNCHANGED)
        test_rgb_image = np.stack([test_rgb_image[...,-1],test_rgb_image[...,-2],test_rgb_image[...,-3]], axis=-1)
        test_d_image = imread(self.config["data_path"] + Config.exampleDImageName, IMREAD_ANYDEPTH)
                                              
        priors = PriorDecl.prepare_prior(PriorDecl.examplePrior)
        out = self(test_rgb_image, test_d_image, priors=priors, lookAt_t=lookAt_t)
        
        self.config = config
        return out
        