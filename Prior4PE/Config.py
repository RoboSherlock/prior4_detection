import numpy as np

initial_config = {
    "data_path": "",
    "seg_model_name": "models/bb_pred_model_with960x1280",
    "pose_model_name": "models/pred_prio_model_fixed_zcorr_nosegin__",

    "pei_xyDim": 96,
    "outputStrides": 2,
    
    "verbose": 0,
    
    "MeanShift_bandwidth": 5
}

#TestValues
exampleRGBImageName = f'test_images/rgb_image_full.png'
exampleDImageName   = f'test_images/depth_image_full.png'

exampleCameraMatrix = np.array([525., 0.0, 319.75, 0.0, 525., 239.75, 0.0, 0.0, 1.0]).reshape((1,3,3))

#ModelInfo (because somhow it is not correctly saved inside the tensorflow model, should be removed in future)
model_infos = {
    1: {'diameter': 263.129667788082,
        'mins': np.array([-131.5648, -131.5648,    0.    ]),
        'maxs': np.array([131.5648 , 131.5647 ,  19.08008]),
        'symmetries_discrete': [],
        'symmetries_continuous': True
       },
    2: {'diameter': 196.66194030304285,
        'mins': np.array([-98.33089, -98.33094,   0.     ]),
        'maxs': np.array([98.33087, 98.33092, 19.48219]),
        'symmetries_discrete': [],
        'symmetries_continuous': True
       },
    3: {'diameter': 169.9998709097113,
        'mins': np.array([-39.99037, -40.00001,   0.     ]),
        'maxs': np.array([ 39.99036,  40.00001, 150.     ]),
        'symmetries_discrete': [np.array([[ 7.3887e-15, -1.0000e+00,  0.0000e+00,  0.0000e+00],
                                          [ 1.0000e+00,  7.3887e-15,  0.0000e+00,  0.0000e+00],
                                          [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00],
                                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
                                         ]),
                                np.array([[-1.00000e+00, -6.05186e-14,  0.00000e+00,  0.00000e+00],
                                          [ 6.05186e-14, -1.00000e+00,  0.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
                                         ]),
                                np.array([[-2.86054e-14,  1.00000e+00,  0.00000e+00,  0.00000e+00],
                                          [-1.00000e+00, -2.86054e-14,  0.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
                                         ])
                               ],
        'symmetries_continuous': False
       },
    4: {'diameter': 169.9998709097113,
        'mins': np.array([-39.99037, -40.00001,   0.     ]),
        'maxs': np.array([ 39.99036,  40.00001, 150.     ]),
        'symmetries_discrete': [np.array([[ 7.3887e-15, -1.0000e+00,  0.0000e+00,  0.0000e+00],
                                          [ 1.0000e+00,  7.3887e-15,  0.0000e+00,  0.0000e+00],
                                          [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00],
                                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
                                         ]),
                                np.array([[-1.00000e+00, -6.05186e-14,  0.00000e+00,  0.00000e+00],
                                          [ 6.05186e-14, -1.00000e+00,  0.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
                                         ]),
                                np.array([[-2.86054e-14,  1.00000e+00,  0.00000e+00,  0.00000e+00],
                                          [-1.00000e+00, -2.86054e-14,  0.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
                                          [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
                                         ])
                               ],
        'symmetries_continuous': False
       }
}
