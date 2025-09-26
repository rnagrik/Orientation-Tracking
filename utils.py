import time
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np


def tic():
    return time.time()

def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_dataset(dataset):

    imu_datasets = [1,2,3,4,5,6,7,8,9,10,11]
    vic_datasets = [1,2,3,4,5,6,7,8,9]
    cam_datasets = [1,2,8,9,10,11]

    assert dataset in imu_datasets, f"Dataset {dataset} not available. Choose from {imu_datasets}"
    print(f"Reading dataset {dataset}...")

    ts = tic()
    def read_f(fname):
        with open(fname, 'rb') as f:
            if sys.version_info[0] < 3:
                d = pickle.load(f)
            else:
                d = pickle.load(f, encoding='latin1')  # needed for python 3
        return d


    imud = read_f("data/imu/imuRaw" + str(dataset) + ".p")
    print("IMU data read...")
    
    camd = None
    vicd = None

    if dataset in cam_datasets:
        camd = read_f("data/cam/cam" + str(dataset) + ".p")
        print("Camera data read...")
    else:
        print("No camera data for this dataset")

    if dataset in vic_datasets:
        vicd = read_f("data/vicon/viconRot" + str(dataset) + ".p")
        print("Vicon data read...")
    else:
        print("No Vicon data for this dataset")

    toc(ts, "Reading data")
    
    return imud, camd, vicd



def process_imu_data(imud):
    Vref = 3300 #mV
    g = 9.81
    a_scale = Vref/(1023*330)
    w_scale = Vref/1023/3.33*(np.pi/180)

    # load data and convert to correct format
    imu = imud['vals']
    imu_ts = imud['ts']

    # orig_imu_ids ['-ax','-ay','az','wz','wx','wy']
    # corr_imu_ids [ 'ax', 'ay','az','wx','wy','wz']

    imu = imu.astype(int)
    imu[:2,:] = -imu[:2,:]
    wz = np.copy(imu[3,:])
    imu[3,:],imu[4,:],imu[5,:] = imu[4,:],imu[5,:],wz
    size = imu.shape[1]

    # bias, az, scaling corrections by taking mean of first 200 samples
    bias = np.mean(imu[:,:200],axis = 1).reshape((6,1))
    corrected_imu = imu - bias
    corrected_imu[:3,:] *= a_scale
    corrected_imu[2,:]  += 1
    corrected_imu[:3,:] *= (-g)
    corrected_imu[3:,:] *= w_scale

    return corrected_imu, imu_ts


def PlotIMUvOPTaccn(imu_data,opt_accn_data):
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(imu_data[0,:],label="IMU ax")
    plt.plot(opt_accn_data[:,0],label="OPT ax")
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(imu_data[1,:],label="IMU ay")
    plt.plot(opt_accn_data[:,1],label="OPT ay")
    plt.legend()
    plt.title("IMU vs. Optimised Accelerations")
    plt.subplot(1,3,3)
    plt.plot(imu_data[2,:],label="IMU az")
    plt.plot(opt_accn_data[:,2],label="OPT az")
    plt.legend()
    plt.show(block=False)

