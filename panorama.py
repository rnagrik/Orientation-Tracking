import numpy as np
import matplotlib.pyplot as plt


class Panorama:

    def __init__(self,Images,Image_ts,RotMat,RotMat_ts):
        self.a = 2
        self.phi_range = np.pi/3                                     # horizontal FoV of the camera
        self.lambda_range = np.pi/4                                  # vertical FoV of the camera                      
        self.ImageSize = np.array([Images.shape[0],Images.shape[1]])  
        self.Images = Images
        self.Image_ts = Image_ts
        self.RotMat = RotMat
        self.RotMat_ts = RotMat_ts
        self.PanoImgSize = [720,1080]
        self.SyncTimeStamps()
    
    def SyncTimeStamps(self):
        # Sync Image timestamps with Rotation Matrix timestamps
        print()
        print("Image size: ", self.Images.shape)
        print("Image Tstamps: ", self.Image_ts.shape)
        print("RotMat size: ", self.RotMat.shape)
        print("RotMat Tstamps: ", self.RotMat_ts.shape)

        AdjustedRotMat = np.zeros((3,3,self.Image_ts.shape[1]))
        AdjustedRotMatTimestamps = np.zeros((1,self.Image_ts.shape[1]))

        i,j = 0,0 # i = Image_counter, j = RotMat_counter
        while i < self.Image_ts.shape[1] and j != self.RotMat_ts.shape[1]:
            if self.Image_ts[0,i] > self.RotMat_ts[0,j]:
                j += 1
            else:
                forward_diff = abs(self.Image_ts[0,i] - self.RotMat_ts[0,j])
                backward_diff = abs(self.Image_ts[0,i] - self.RotMat_ts[0,j-1])    
                if min(forward_diff,backward_diff) == backward_diff:
                    j -= 1
                AdjustedRotMat[:,:,i] = self.RotMat[:,:,j]
                AdjustedRotMatTimestamps[0,i] = self.RotMat_ts[0,j]
                j += 1
                i += 1
        
        self.RotMat,self.RotMat_ts = AdjustedRotMat[:,:,:i], AdjustedRotMatTimestamps[:,:i]
        self.Images,self.Image_ts = self.Images[:,:,:,:i], self.Image_ts[:,:i]
        print("")
        print("new Image size: ",self.Images.shape)
        print("new Image Tstamps: ", self.Image_ts.shape)
        print("new RotMat size: ", self.RotMat.shape)
        print("new RotMat Tstamps: ", self.RotMat_ts.shape)
 

    def Pixel2Spherical_Transform(self):
        # From pixel coordinates (u,v) to spherical coordinates (phi, lambda)
        H,W = self.ImageSize[0],self.ImageSize[1]
        p_i,p_j = np.arange(W),np.arange(H)
        Pxl2Sphr_Transform_phi = np.tile(p_i,(H,1))*self.phi_range/(W-1) - self.phi_range/2
        Pxl2Sphr_Transform_lambda = np.tile(p_j,(W,1)).T*self.lambda_range/(H-1) - self.lambda_range/2

        Phi_Lambda = np.concatenate(([Pxl2Sphr_Transform_phi],[Pxl2Sphr_Transform_lambda]),axis=0)
        return Phi_Lambda

    def Spherical2UnitCartesian_Transform(self,PhiLamMat):
        phi = PhiLamMat[0] 
        lam = PhiLamMat[1]

        # x y z - Camera Frame
        x_cam = np.sin(phi)*np.cos(lam)
        y_cam = np.sin(lam)
        z_cam = np.cos(phi)*np.cos(lam)

        # X Y Z - Body Frame
        X = z_cam
        Y = -x_cam
        Z = -y_cam
        
        XYZ = np.concatenate(([X],[Y],[Z]),axis=0)
        return XYZ

    def Cart2WorldCartforAllTime(self,Sphr2CartMat):

        all_rotmat = np.transpose(self.RotMat,(2,0,1))
        Cart = np.matmul(all_rotmat,Sphr2CartMat.reshape((3,-1)))
        Cart = Cart.reshape((self.RotMat.shape[2],3,Sphr2CartMat.shape[1],Sphr2CartMat.shape[2]))
        WorldCart = np.transpose(Cart,(0,2,3,1))

        return WorldCart

    def WorldCart2WorldSphr(self,AllWorldCart):
        x = AllWorldCart[:,:,:,0]
        y = AllWorldCart[:,:,:,1]
        z = AllWorldCart[:,:,:,2]
        r = np.sqrt(x**2+y**2+z**2) + 0.0001

        phi = np.arctan2(-y,x)
        lam = np.arcsin(-z/r)

        PhiLam = np.concatenate(([phi],[lam]),axis=0)
        PhiLam = np.transpose(PhiLam,(1,2,3,0))

        return PhiLam

    def Spherical2Pixel(self,PhiLam):

        H,W = self.PanoImgSize[0],self.PanoImgSize[1]
        pano_phi_range = 2*np.pi
        pano_lam_range = np.pi

        phi = PhiLam[:,:,:,0]
        lam = PhiLam[:,:,:,1]
        
        Pixel_i = ((2*phi+pano_phi_range)/(2*pano_phi_range))*(W-1)
        Pixel_j = ((2*lam+pano_lam_range)/(2*pano_lam_range))*(H-1)

        Pixel_ij = np.concatenate(([Pixel_i],[Pixel_j]),axis=0)
        Pixel_ij = np.transpose(Pixel_ij,(1,2,3,0))
        Pixel_ij = Pixel_ij.astype(int)

        return Pixel_ij

    def AllPixel2Image(self,Pixels_data):

        ImageRGB = np.transpose(self.Images,(3,0,1,2))
        FinalImage = np.zeros((self.PanoImgSize[0],self.PanoImgSize[1],3), dtype=np.uint8)
        FinalImage[Pixels_data[:,:,:,1],Pixels_data[:,:,:,0]] = ImageRGB[:,:,:]
        print("Panorama Generated...")

        return FinalImage

    def StitchImage(self):
        print("Generating Panorama...")
        Pxl2Sphr_Transform = self.Pixel2Spherical_Transform()
        Sphr2Cart_Transform = self.Spherical2UnitCartesian_Transform(Pxl2Sphr_Transform)
        AllWorldCart = self.Cart2WorldCartforAllTime(Sphr2Cart_Transform)
        AllWorldSphr = self.WorldCart2WorldSphr(AllWorldCart)
        Pixels = self.Spherical2Pixel(AllWorldSphr)
        FinalStitchedImage = self.AllPixel2Image(Pixels)
        plt.figure()
        plt.imshow(FinalStitchedImage)
        plt.show(block=False)
        