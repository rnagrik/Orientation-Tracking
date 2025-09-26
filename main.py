import sys
import argparse
import numpy as np
from optimizer import Optimizer
from panorama import Panorama
from utils import *
from model import *
	

def main():
	#---------------------------Data Read/Process---------------------------
	imud, camd, vicd = read_dataset(args.dataset)
	imu, imu_ts = process_imu_data(imud)
   
	#---------------------------Orientation Initial Estimation---------------------------
	Qmatrix = getQuatsFromMotionModel(imu, imu_ts)

	#----------------------------Set up Optimization---------------------------------

	all_touts = np.diff(imu_ts)[0,:Qmatrix.shape[0]]    # time intervals (tou) from imu data
	all_wts = imu[3:,:].T 								# angular velocities from imu data
	Amatrix = imu[:3,:Qmatrix.shape[0]].T 				# acceleration from imu data

	Opt = Optimizer(Amatrix,Qmatrix,all_touts,all_wts,alpha=args.alpha,max_iter=args.max_iter)
	Opt.OptimizeQ()
	Opt.evalRotationMatrices()
	all_OPT_ats = CalculateAccelerationFromQuats(Opt.Q)

	Opt.PlotErrorvsIter()
	PlotRPY_IMUvsOPTvsGT(Opt.Q_initial,Opt.Q,imu_ts,vicd)
	PlotIMUvOPTaccn(imu,all_OPT_ats)

	#--------------------------Generate Panorama------------------------------
	
	if camd is not None:
		Image = camd['cam']
		Image_ts = camd['ts']
		OPTRotMat = Opt.RotMatrices
		OPTRotMat_ts = imud['ts']
		a = Panorama(Image,Image_ts,OPTRotMat,OPTRotMat_ts)
		a.StitchImage()
	else:
		print("No camera data available for this dataset to generate panorama")

	input("Plots Generated. Press Enter to exit...")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--dataset", type=int, default=10, help="Dataset number (default: 10)")
	parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for optimizer (default: 100)")
	parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate for optimizer (default: 0.001)")
	args = parser.parse_args()

	main()

