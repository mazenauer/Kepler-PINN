import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from benchmark import benchmark
from scipy.optimize import fsolve

class State:
    
    def __init__(self,q,p):
        self.q = np.asarray(q,dtype=np.float64)
        self.p = np.asarray(p,dtype=np.float64)
        self.x = np.array([self.q[0]])
        self.y = np.array([self.q[1]])
        self.px = np.array([self.p[0]])
        self.py = np.array([self.p[1]])
        self.t = np.array([0])
        self.H = np.array([self.Hamiltonian()])
        self.L = np.array([self.angularMomentum()])
        
    def showState(self):
        x,y = self.q
        plt.scatter([0,x],[0,y])
        plt.xlim(-abs(x)*1.1,abs(x)*1.1)
        plt.ylim(-abs(y)*1.1,abs(y)*1.1)
        plt.show()
        
    def showTrace(self,marker='b-'):
        fig,ax = plt.subplots(figsize=(8,8))
        ax.set_aspect(1)
        plt.plot(self.x,self.y,marker)
        plt.scatter([0],[0],color="red")
        plt.xlim(-np.max(np.abs(self.x))*1.1,np.max(np.abs(self.x))*1.1)
        plt.ylim(-np.max(np.abs(self.y))*1.1,np.max(np.abs(self.y))*1.1)
        plt.show()
        
    def showStats(self):
        if len(self.H) == 0:
            raise "No stats saved"
        fig,ax = plt.subplots(1,2,figsize=(8,6))
        ax[0].plot(self.t,self.H,'b-')
        ax[1].plot(self.t,self.L,'g-')
        plt.show()
    
    def qdot(self,p):
        return p
    
    def pdot(self,q):
        return -q/np.linalg.norm(q)**3
    
    def Hamiltonian(self):
        return 0.5*np.linalg.norm(self.p)**2-1/np.linalg.norm(self.q)
    
    def angularMomentum(self):
        return self.q[1]*self.p[0]-self.q[0]*self.p[1]
    
    def symplecticEulerStep(self,dt):
        self.q += dt*self.qdot(self.p)
        self.p += dt*self.pdot(self.q)
        
    def impTrapStep(self,dt):
        def hfunc(x):
            return np.array([x[0:2]-self.q-0.5*dt*(self.qdot(self.p)+self.qdot(x[2:])), x[2:]-self.p-0.5*dt*(self.pdot(self.q)+self.pdot(x[:2]))]).ravel()
        x = np.array([self.q,self.p]).ravel()
        xnew = fsolve(hfunc,x)
        self.q = xnew[:2]
        self.p = xnew[2:]
    
    def explicitEulerStep(self,dt):
        q = self.q
        p = self.p
        self.q = q+dt*self.qdot(p)
        self.p = p+dt*self.pdot(q)
        
    def RK4Step(self,dt):
        q,p = self.q,self.p
        k1,l1 = self.qdot(p),self.pdot(q)
        k2,l2 = self.qdot(p+0.5*dt*l1),self.pdot(q+0.5*dt*k1)
        k3,l3 = self.qdot(p+0.5*dt*l2),self.pdot(q+0.5*dt*k2)
        k4,l4 = self.qdot(p+dt*l3),self.pdot(q+dt*k3)
        self.q += dt/6*(k1+2*k2+2*k3+k4)
        self.p += dt/6*(l1+2*l2+2*l3+l4)
        
    def symmetricRuthStep(self,dt):
        b = [7/48,3/8,-1/48,-1/48,3/8,7/48]
        bhat = [1/3,-1/3,1,-1/3,1/3,0]
        n = len(b)
        for i in range(n):
            self.p += b[i]*dt*self.pdot(self.q)
            self.q += bhat[i]*dt*self.qdot(self.p)
    
    def evolve(self,dt,Tmax,saveStats=True,saveNthStep=False,solver=None):
        t = 0
        if saveNthStep==False:
            saveNthStep = 1
        if solver==None:
            solver = self.symmetricRuthStep
        if saveStats:
            while t-dt<=Tmax:
                for i in range(saveNthStep):
                    solver(dt)
                    t += dt
                self.x = np.append(self.x,self.q[0])
                self.y = np.append(self.y,self.q[1])
                self.px = np.append(self.px,self.p[0])
                self.py = np.append(self.py,self.p[1])
                self.t = np.append(self.t,t)
                self.H = np.append(self.H,self.Hamiltonian())
                self.L = np.append(self.L,self.angularMomentum())
             
            return self.t,self.x,self.y,self.H,self.L
        
        else:
            while t+dt<=Tmax:
                solver(dt)
            return
                
    def getFunctions(self):
        x = interp1d(self.t,self.x,kind="cubic")
        y = interp1d(self.t,self.y,kind="cubic")
        px = interp1d(self.t,self.px,kind="cubic")
        py = interp1d(self.t,self.py,kind="cubic")
        return x,px,y,py