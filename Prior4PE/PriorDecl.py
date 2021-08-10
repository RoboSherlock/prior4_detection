from scipy.spatial.transform import Rotation
import numpy as np

examplePrior = {
    "countOfPriors": 1,
    "informationPerPrior": 3
}

examplePrior.update({
    "planeRotationName": None,
    "planeTranslationName": None,
    
    "planeRotation": examplePrior["informationPerPrior"] * [Rotation.from_quat([-0.55142, -0.391561, 0.660581,   0.325942])], #Attention: scalar-last format,
    "planeTranslation": examplePrior["informationPerPrior"] * [np.array([-0.0457627, -0.217613, 1.23532])],

    "point":  np.array([0, 0, 0,   0, 0, 0,   0, 0, 100.]).reshape((1, examplePrior["countOfPriors"], examplePrior["informationPerPrior"], 3)),
    "pointd": np.array([   0,         0,         0.     ]).reshape((1, examplePrior["countOfPriors"], examplePrior["informationPerPrior"],  )),
    "priorw": np.array([ 1./30.,   1./100.,    1./100.  ]).reshape((1, examplePrior["countOfPriors"], examplePrior["informationPerPrior"],  ))
})

def calculatePlane_x_d(list_Rt, _Dir = [np.array([0,0,1.])]):
    if len(_Dir) == 1:
        _Dir *= len(list_Rt)
        
    plane_xs = []
    plane_ds = []
    for i, Rt in enumerate(list_Rt):
        R, t = Rt
        plane_x = R.as_matrix() @ _Dir[i]
        plane_d = plane_x @ t
        plane_xs.append(plane_x)
        plane_ds.append(plane_d)
    return plane_xs, plane_ds

def prepare_prior(priorConfig):
    dw_x, dw_d = calculatePlane_x_d(list(zip(priorConfig["planeRotation"], [_* 1000 for _ in priorConfig["planeTranslation"]])))

    planeX = np.array(dw_x).reshape((1, priorConfig["countOfPriors"], priorConfig["informationPerPrior"], 3))
    planeD = np.array(dw_d).reshape((1, priorConfig["countOfPriors"], priorConfig["informationPerPrior"]   ))

    return planeX, planeD, priorConfig["point"], priorConfig["pointd"], priorConfig["priorw"]

    
