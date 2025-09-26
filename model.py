import numpy as np
from matplotlib import pyplot as plt
from quaternion import Quaternion
import transforms3d 

G = 9.81  # gravity

def getQuatsFromMotionModel(imu, imu_ts):
    # get all q_t's from imu data
    q0 = [1,0,0,0]
    qt = Quaternion(q0)
    Qmatrix = np.array(q0).reshape((1,4))
    size = imu.shape[1]
    
    for i in range(size-1):
        wt = imu[3:,i]
        [a,b,c] = (imu_ts[0,i+1]-imu_ts[0,i])*wt/2
        q2 = Quaternion([0,a,b,c])
        q2 = q2.getExp()
        qt = Quaternion.QuatMultiply(qt,q2)
        Qmatrix = np.vstack((Qmatrix, np.array(qt.array).reshape((1,4))))
    Qmatrix = Qmatrix[1:]
    print("----------- All q_t's estimated------------")
    print("Quatmatrix size - ", Qmatrix.shape)

    return Qmatrix

def CalculateAccelerationFromQuats(Qmatrix):
    # get all a_t's from q_t's
    q = Quaternion([0,0,0,-G])
    Amatrix = np.array([0,0,0]).reshape((1,3))

    for i in range (Qmatrix.shape[0]):
        qt = Quaternion(Qmatrix[i])
        qt_inv = qt.getInverse()
        at = Quaternion.QuatMultiply(Quaternion.QuatMultiply(qt_inv,q),qt)
        Amatrix = np.vstack((Amatrix,np.array(at.array[1:]).reshape((1,3))))

    Amatrix = Amatrix[1:]
    print("----------- All a_t's estimated------------")
    print("Amatrix size - ", Amatrix.shape)
    return Amatrix

def GetRollPitchYaw(quat_list,GTRotMat=None):

    EstimatedRollPitchYaw = np.array([0,0,0]).reshape((1,3))
    for quat in quat_list:
        RollPitchYaw = transforms3d.euler.quat2euler(quat, axes='sxyz')
        EstimatedRollPitchYaw = np.vstack((EstimatedRollPitchYaw,np.array(RollPitchYaw).reshape((1,3))))
    EstimatedRollPitchYaw = EstimatedRollPitchYaw[1:]
    
    GTRollPitchYaw = None
    if GTRotMat is not None:
        GTRollPitchYaw = np.array([0,0,0]).reshape((1,3))
        for i in range(1,GTRotMat.shape[2]):
            RollPitchYaw = transforms3d.euler.mat2euler(GTRotMat[:,:,i],axes='sxyz')
            GTRollPitchYaw = np.vstack((GTRollPitchYaw,np.array(RollPitchYaw).reshape((1,3))))
        GTRollPitchYaw = GTRollPitchYaw[1:]
        
    return EstimatedRollPitchYaw, GTRollPitchYaw

def PlotRPY_IMUvsOPTvsGT(IMU_quats,OPT_quats,imu_ts,vicon_data=None):

    if vicon_data is not None:
        vic_ts = vicon_data['ts']
        vic_R = vicon_data['rots']
    else:
        vic_ts = None
        vic_R = None
  
    IMU_RPY, GroundTruthRPY = GetRollPitchYaw(IMU_quats,vic_R)
    OPT_RPY, GroundTruthRPY = GetRollPitchYaw(OPT_quats,vic_R)

    IMU_time = imu_ts[0,:IMU_RPY.shape[0]]
    if vicon_data is not None:
        VIC_time = vic_ts[0,:GroundTruthRPY.shape[0]]

    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(IMU_time,IMU_RPY[:,0], label="Calibrated IMU Roll")
    plt.plot(IMU_time,OPT_RPY[:,0], label="Optimised Roll")
    if vicon_data is not None:
        plt.plot(VIC_time,GroundTruthRPY[:,0], label="GroundTruth Roll")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(IMU_time,IMU_RPY[:,1], label="Calibrated IMU Pitch")
    plt.plot(IMU_time,OPT_RPY[:,1], label="Optimised Pitch")
    if vicon_data is not None:
        plt.plot(VIC_time,GroundTruthRPY[:,1], label="GroundTruth Pitch")
    plt.legend()
    plt.title("IMU_calibrated vs Optimised vs GroundTruth RPY")

    plt.subplot(1, 3, 3)  
    plt.plot(IMU_time,IMU_RPY[:,2], label="Calibrated IMU Yaw")
    plt.plot(IMU_time,OPT_RPY[:,2], label="Optimised Yaw")
    if vicon_data is not None:
        plt.plot(VIC_time,GroundTruthRPY[:,2], label="GroundTruth Yaw")
    plt.legend()

    plt.show(block=False)



