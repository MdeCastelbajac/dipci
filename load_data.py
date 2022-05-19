import numpy as np
from utils import normalize, downsample

def loadDataR09( upscale_factor ):
    
    PATH = "./Data"
    data_ssh = PATH + "/1308_square_NATL60_SSH_R09.npy"
    data_sst = PATH + "/1308_square_NATL60_SST_R09.npy"
    ssh = np.load(data_ssh)
    sst = np.load(data_sst)
    ssh_norm = np.array([normalize(img,0,1) for img in ssh])
    sst_norm = np.array([normalize(img,0,1) for img in sst])
    ssh_lr = np.array(
            [downsample(img, upscale_factor) for img in ssh_norm]
            )
    sst_lr = np.array(
            [downsample(img, upscale_factor//2) for img in sst_norm]
            )
    return ssh, sst, ssh_norm, sst_norm, ssh_lr, sst_lr

def loadDataR18( upscale_factor ):

    PATH = "./Data"
    data_ssh = PATH + "/1308_square_NATL60_SSH_R09.npy"
    data_sst = PATH + "/1308_square_NATL60_SST_R09.npy"
    ssh = np.load(data_ssh)
    sst = np.load(data_sst)
    ssh = np.array( 
            [downsample(img, upscale_factor//2) for img in ssh]
            )
    sst = np.array( 
            [downsample(img, upscale_factor//2) for img in sst]
            )
    ssh_norm = np.array([normalize(img,0,1) for img in ssh])
    sst_norm = np.array([normalize(img,0,1) for img in sst])
    ssh_lr = np.array(
            [downsample(img, upscale_factor) for img in ssh_norm]
            )
    sst_lr = np.array(
            [downsample(img, upscale_factor//2) for img in sst_norm]
            )
    return ssh, sst, ssh_norm, sst_norm, ssh_lr, sst_lr

