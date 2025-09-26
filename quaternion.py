import numpy as np

class Quaternion:
    # a = [qs,qv1,qv2,qv3]

    def __init__(self,a:list):
        self.s = a[0]
        self.v = np.array(a[1:])
        self.array = a
        self.norm = np.sqrt(self.s**2 + np.dot(self.v,self.v))

    def normalize(self) -> None:
        self.norm = np.sqrt(self.s**2 + np.dot(self.v,self.v))
        self.s = self.s/self.norm
        self.v = self.v/self.norm
        self.norm = 1
    
    def getInverse(self):
        # q^-1 = q*/||q||^2
        i_s = self.s/(self.norm)**2
        i_v = -self.v/(self.norm)**2
        i_quat = [i_s,i_v[0],i_v[1],i_v[2]]
        return Quaternion(i_quat) 
    
    def getConjucate(self):
        # q* = [qs,-qv]
        c_v = -self.v
        c_quat = [self.s,c_v[0],c_v[1],c_v[2]]
        return Quaternion(c_quat)
    
    def getExp(self):
        # exp(q) = [exp(qs)*cos(||qv||), exp(qs)*sin(||qv||)*qv/||qv||]
        norm_v = np.linalg.norm(self.v)
        q1 = np.exp(self.s)*np.cos(norm_v)
        q2 = np.exp(self.s)*np.sin(norm_v)*self.v/norm_v
        expquat = np.array([q1,q2[0],q2[1],q2[2]])
        return Quaternion(expquat)
    
    def getLog(self):
        # log(q) = [log(||q||), arccos(qs/||q||)*qv/||qv||]
        norm_v = np.linalg.norm(self.v)
        q1 = np.log(self.norm)
        q2 = (np.arccos(self.qs/self.norm))
        logquat = np.array([q1,q2*self.v/norm_v])
        return Quaternion(logquat)
    
    def getRotMat(self):
        # R = (||q||^2)^-1 * (E^T)*G, E = [-qv, qs*I + qv_hat], G = [-qv, qs*I - qv_hat]
        qs = np.copy(self.s)
        qv = np.copy(self.v).reshape([3,1])
        [[x1],[x2],[x3]] = qv
        I = np.eye(3)
        qv_hat = np.array([[0,-x3,x2],[x3,0,-x1],[-x2,x1,0]])
        E = np.hstack(-qv, qs*I + qv_hat)
        G = np.hstack(-qv, qs*I - qv_hat)
        R = E.T.dot(G)
        R = R/(self.norm)**2
        return R

    @staticmethod
    def QuatMultiply(p,q):
        s1 = q.s*p.s - q.v.T.dot(p.v)
        s2 = q.s*p.v + p.s*q.v + np.cross(p.v,q.v)
        newquat = np.array([s1,s2[0],s2[1],s2[2]])
        return Quaternion(newquat)
    
    @staticmethod
    def QuatAdd(p,q):
        qs = p.s + q.s
        qv = q.v + p.v
        addquat = [qs,qv[0],qv[1],qv[2]]
        return Quaternion(addquat)
