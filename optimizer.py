import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import transforms3d

# A = [[qs1,qv11,qv12,qv13],[qs2,qv22,qv22,qv23],...]
def CalculateSigmaNormSquared(A):
    return jnp.trace(jnp.matmul(A,A.T))


class JaxQuat:
    # Parallelized version of quaternion exponential, logarithm, inverse and multiplication operations
    @staticmethod
    def QuatExp(A):
        if A.shape[1] != 4:
            print("QuatMatrix not in correct format")
            raise Exception
        e_qs = jnp.exp(A[:,0])
        normqv = jnp.linalg.norm(A[:,1:],axis=1)+0.001
        cosqv = jnp.cos(normqv)
        sinqv = jnp.sin(normqv)
        expA = jnp.zeros(A.shape)
        expA = expA.at[:,0].set(jnp.multiply(e_qs,cosqv))
        v = jnp.multiply(jnp.divide(e_qs,normqv),sinqv)
        expA = expA.at[:,1:].set(jnp.multiply(v,A[:,1:].T).T)

        return expA

    @staticmethod
    def QuatLog(A):
        normq = jnp.linalg.norm(A,axis=1) + 0.001
        normqv = jnp.linalg.norm(A[:,1:],axis=1) + 0.001
        arccosqsbyq = jnp.arccos(jnp.divide(A[:,0],normq))
        logA = jnp.zeros(A.shape)
        logA = logA.at[:,0].set(jnp.log(normq))
        v = jnp.divide(arccosqsbyq,normqv)
        logA = logA.at[:,1:].set(jnp.multiply(v,A[:,1:].T).T)

        return logA

    @staticmethod
    def QuatInverse(A):
        QuatInv = jnp.copy(A)
        normq = jnp.linalg.norm(A,axis=1).reshape((A.shape[0],1)) + 0.001
        QuatInv = QuatInv.at[:,1:].set(-QuatInv[:,1:])
        QuatInv = jnp.divide(jnp.divide(QuatInv,normq),normq)

        return QuatInv

    @staticmethod
    def QuatMul(A,B):     # A = [[aqs1,aqv11,aqv12,aqv13],...], B = [[bqs1,bqv11,bqv12,bqv13],...]
        if A.shape != B.shape:
            raise Exception("Matrix Dimensions don't match.")
        AB = jnp.zeros(A.shape)
        AB = AB.at[:,0].set(jnp.multiply(A[:,0],B[:,0]) - jnp.sum(jnp.multiply(A[:,1:],B[:,1:]),axis=1))
        AB = AB.at[:,1:].set(jnp.multiply(B[:,1:].T,jnp.tile(A[:,0],(3,1))).T + jnp.multiply(A[:,1:].T,jnp.tile(B[:,0],(3,1))).T + jnp.cross(A[:,1:],B[:,1:]))

        return AB
    
    def __init__(self):
        pass


#-----------------------------------------------------------------

class Optimizer(JaxQuat):
    def __init__(self,Amatrix,Qmatrix,all_touts,all_wts,alpha=1e-3,max_iter=100,err_conv=1e-3):
        self.ats = jnp.array(Amatrix)
        self.Q_initial = jnp.array(Qmatrix)
        self.Q = jnp.array(Qmatrix) # 4xN matrix [[q1],[q2],....,[qT]]
        self.size = self.Q.shape
        self.exptwbytwo = self.ExpTouWts(all_touts,all_wts)
        self.gquat = jnp.tile(jnp.array([[0.0,0.0,0.0,-9.81]]),(self.size[0],1))

        self.error = self.CostFunction(self.Q)
        print(f"Initial error: {self.error:.3f}")

        self.all_iterations_error = [self.error]
        self.err_conv = err_conv
        self.max_iter = max_iter
        self.alpha = alpha
        self.RotMatrices = None

    def ExpTouWts(self,touts,wts):
        w = jnp.array(wts).reshape(max(wts.shape),3)[:self.size[0]]
        t = jnp.array(touts).reshape(max(touts.shape),1)
        t = jnp.tile(t,(1,3))[:self.size[0]]
        wtbytwo = jnp.array(t)*jnp.array(w)/2
        zerowtbytwo = jnp.hstack((jnp.zeros((self.size[0],1)),wtbytwo))

        return self.QuatExp(zerowtbytwo)

    def CostFunction(self,Q):
        qt_inverse = self.QuatInverse(Q)
        f = self.QuatMul(jnp.vstack((jnp.array([[1.0,0.0,0.0,0.0]]),Q))[:-1], self.exptwbytwo)
        qtf= self.QuatMul(qt_inverse,f)
        logqinvf = self.QuatLog(qtf)
        cost1 = 2 * CalculateSigmaNormSquared(logqinvf)
        h_qt = self.QuatMul(self.QuatMul(qt_inverse,self.gquat),Q)
        cost2 = (0.5) * CalculateSigmaNormSquared(self.ats-h_qt[:,1:])

        return cost1 + cost2
    
    def OptimizeQ(self):
        grad_f = jax.grad(self.CostFunction)
        iter = 0
        conv_counter = 0
        while (iter < self.max_iter) and (conv_counter < 5):
            Q_next = self.Q - self.alpha*grad_f(self.Q)
            modQ = (jnp.linalg.norm(Q_next,axis=1)).reshape(Q_next.shape[0],1) + 0.001 # to avoid float division by zero
            self.Q = Q_next/modQ
            error = self.CostFunction(self.Q)
            if abs(1-error/self.error) <= self.err_conv:
                conv_counter += 1
                conv_str = "error converging"
            else:
                conv_counter = 0
                conv_str = ""
            print(f"iter_count: {iter+1} Error: {error} {conv_str}")
            self.error = error
            self.all_iterations_error.append(self.error)
            iter += 1

        if iter == self.max_iter and conv_counter < 5:
            print("Max iterations reached, stopping optimization.")
        else:
            print("Error converged, stopping optimization.")
        
    def PlotErrorvsIter(self):
        plt.figure()
        plt.title("Iterations vs. Error (Cost Function)")
        plt.plot(self.all_iterations_error)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.show(block=False)

    def PlotInitialQvsOptimisedQ(self):
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].plot(self.Q_initial[:,0],label="imu q1")
        axs[0, 0].plot(self.Q[:,0],label="optimised q1")

        axs[0, 1].plot(self.Q_initial[:,1],label="imu q2")
        axs[0, 1].plot(self.Q[:,1],label="optimised q2")

        axs[1, 0].plot(self.Q_initial[:,2],label="imu q3")
        axs[1, 0].plot(self.Q[:,2],label="optimised q3")

        axs[1, 1].plot(self.Q_initial[:,3],label="imu q4")
        axs[1, 1].plot(self.Q[:,3],label="optimised q4")

        plt.legend()
        plt.show()

    def evalRotationMatrices(self):
        # self.RotMatrices = [R1,R2,...,RT] where Ri is 3x3 rotation matrix at time i
        Rot = transforms3d.euler.quat2mat(self.Q[0]).reshape((1,3,3))
        for quat in self.Q[1:]:
            RMatrix = transforms3d.euler.quat2mat(quat).reshape((1,3,3))
            Rot = np.concatenate((Rot,RMatrix),axis=0)
        self.RotMatrices = np.transpose(Rot,(1,2,0))
