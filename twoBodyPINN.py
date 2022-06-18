#import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
from odeintegrator import State

if torch.cuda.is_available():
    device = "cuda"
    print("Running on GPU\n")
else:
    device = "cpu"
    print("Running on CPU\n")
    
#set plotting backend
#%matplotlib qt5

#set seed for reproducibility
torch.manual_seed(0);



class NeuralNet(nn.Module):
    """
    Neural net class. Constructor needs input dimension, output dimension, number of hidden layers, and int or list of ints with number of neurons per hidden layer.
    Activation function per default is tanh. Other examples are nn.ReLU(), nn.Sigmoid().
    Methods are "forward" (called as self(x)) for evaluation, "xavier" for weight initialization, "lossFunction" and "fit" are self-explanatory. 
    """
    def __init__(self,input_dimension,output_dimension,n_hidden_layers,neurons,activation_function=nn.Tanh(),initial=torch.tensor([1.,0,0,1]),seed=0):
        super(NeuralNet,self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initial = initial
        if type(neurons)==list:
            assert len(neurons)==n_hidden_layers+1, "Hidden layers and neuron dimension don't match"
            self.neurons = neurons
        else:
            self.neurons = [neurons for _ in range(n_hidden_layers+1)]
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation_function
        self.seed = seed
        
        assert self.n_hidden_layers != 0, "Not a neural network"
        self.input_layer = nn.Linear(self.input_dimension,self.neurons[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i],self.neurons[i+1]) for i in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons[n_hidden_layers],self.output_dimension)
        self.size = np.sum([np.prod([i for i in el.shape]) for el in self.parameters()]) #total number of parameters
        
        self.to(device) #save object to device (either gpu or cpu)

    def forward(self,t):
        t = self.activation(self.input_layer(t))
        for l in self.hidden_layers:
            t = self.activation(l(t))
        return self.output_layer(t)
    
    def conservedQuantities(self,qx,px,qy,py):
        d = (qx.square()+qy.square()).sqrt()#+1e-6
        H = 0.5*(px.square()+py.square())-1/d
        L = qx*py-qy*px
        Ax = (qy*px-qx*py)*py+qx/d
        Ay = (qx*py-qy*px)*px+qy/d
        return H,L,Ax,Ay
    
    def conservedQuantitiesNumpy(self,qx,px,qy,py):
        H,L,Ax,Ay = self.conservedQuantities(torch.tensor(qx),torch.tensor(px),torch.tensor(qy),torch.tensor(py))
        return H.detach().numpy(), L.detach().numpy(), Ax.detach().numpy(), Ay.detach().numpy()
    
    def lossFunction(self,t_train,weights=[1,1,1,0],batch_size=None):
        """
        Perhaps the most important piece of code. Takes the sample training times as a pytorch array, requires property "requires_grad=True".
        The weights give a different weight to the four kinds of losses: ODE loss, initial conditions loss, loss from conservation laws and parameter regularization.
        Returns weighted sum of losses as a real number of type pytorch.tensor. If "batch_size" is given only this fraction of random training samples are used for each epoch.
        """
        #initial time in correct shape
        t0 = torch.tensor([0.],device=device).reshape(-1,1)
        if batch_size != None:
            n = t_train.shape[0]
            t_train = t_train[np.random.choice(n,int(batch_size*n),replace=False)]
        #exact and predicted initial values
        u_exact_0 = self.initial
        u_pred_0 = self(t0)[0].flatten()
        H0,L0,Ax0,Ay0 = self.conservedQuantities(*u_exact_0)
        #predict values of the network at training times and separate them into q and p (postion and momentum)
        u_pred = self(t_train)
        qx_pred = u_pred[:,0]
        px_pred = u_pred[:,1]
        qy_pred = u_pred[:,2]
        py_pred = u_pred[:,3]
        
        #calculate the derivatives of q and p with respect to the training times and keep them stored (create_graph needs True!)
        qx_dot = torch.autograd.grad(qx_pred, t_train, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(qx_pred))[0]
        px_dot = torch.autograd.grad(px_pred, t_train, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(px_pred))[0]
        qy_dot = torch.autograd.grad(qy_pred, t_train, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(qy_pred))[0]
        py_dot = torch.autograd.grad(py_pred, t_train, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(py_pred))[0]
        
        #flatten the arrays to make them one-dimensional
        qx_dot = qx_dot.flatten()
        px_dot = px_dot.flatten()
        qy_dot = qy_dot.flatten()
        py_dot = py_dot.flatten()
        qx_pred = qx_pred.flatten()
        px_pred = px_pred.flatten()
        qy_pred = qy_pred.flatten()
        py_pred = py_pred.flatten()
        t_train = t_train.flatten()

        #ODE loss (two ways: via system of equations or directly the exact ODE)  as sum of mean squared errors
        d3 = (qx_pred.square()+qy_pred.square()).pow(1.5)#+1e-6
        DEloss = (qx_dot-px_pred).square().mean() + (d3*px_dot+qx_pred).square().mean() + (qy_dot-py_pred).square().mean() + (d3*py_dot+qy_pred).square().mean()

        #Initial conditions loss 
        ICloss = (u_pred_0-u_exact_0).square().mean()
        
        #Loss from conserved quantities
        H,L,Ax,Ay = self.conservedQuantities(*u_pred.T)
        CLloss = (H-H0).square().mean() + (L-L0).square().mean() + (Ax-Ax0).square().mean() + (Ay-Ay0).square().mean()

        #parameter regularization
        s,num = 0,0
        for el in self.parameters():
            s += el.square().sum()
            num += el.shape[0]
        Regloss = s/num
        
        return (weights[0]*DEloss+weights[1]*ICloss+weights[2]*CLloss+weights[3]*Regloss)/sum(weights)
        
    def validationLoss(self,xexact,pxexact,yexact,pyexact,t_train):
        u_pred = self(t_train).detach().numpy()
        qx_pred = u_pred[:,0]
        px_pred = u_pred[:,1]
        qy_pred = u_pred[:,2]
        py_pred = u_pred[:,3]
        return 0.5*(L2Error(xexact,yexact,qx_pred,qy_pred)+L2Error(pxexact,pyexact,px_pred,py_pred))
        
    def fit(self,t_train,epochs,verbose=False,weights=[1,1,1,0],optimizer=None,
            batch_size=None,reltol=1e-7,maxtol=1e12,abstol=1e-7,minloss=1e-6):
        """
        t_train is a pytorch array of training points. Epochs is the number of iterations in the fitting process. 
        Verbose prints current epoch and loss. Weights adjusts influence of ODE loss, initial conditions loss, loss from conservation laws, and weight regularization.
        Optimizer is an instance of any of the pytorch.optim classes, for example optim.Adam. If optimizer=None it defaults to Adam.
        Returns the loss history of the training process.
        """
        #history of losses
        history = []
        validationLoss = []
        s = State(self.initial[::2],self.initial[1::2])
        Tmax = t_train.max().item()
        h = (t_train[1]-t_train[0]).item()
        resFactor = 4
        s.evolve(h/resFactor,Tmax)
        xexact = s.x[::resFactor]
        pxexact = s.px[::resFactor]
        yexact = s.y[::resFactor]
        pyexact = s.py[::resFactor]
        #initialize an optimizer instance if none is given
        if optimizer==None:
            optimizer = optim.LBFGS(self.parameters(),lr=0.1)
        #training loop
        oldloss = maxtol
        for i in range(epochs):
            #set all gradients of the NN leaves to zero (needed in pytorch)
            optimizer.zero_grad()
            #calculate current loss
            loss = self.lossFunction(t_train,weights,batch_size)
            validationLoss.append(self.validationLoss(xexact,pxexact,yexact,pyexact,t_train))
            if loss>maxtol:
                flag = "maxtol"
                return history,validationLoss,flag
            if abs(1-loss/oldloss)<reltol:
                flag = "reltol"
                return history,validationLoss,flag
            if abs(oldloss-loss)<abstol:
                flag = "abstol"
                return history,validationLoss,flag
            if abs(loss)<minloss:
                flag = "minloss"
                return history,validationLoss,flag
            oldloss = loss
            #calculate gradient of loss with respect to the NN leaves and store them.
            loss.backward(retain_graph=True)
            #execute one training step given the loss and the gradients that are stored in the NN object.
            optimizer.step(closure = lambda : loss)
            #append current loss to history
            history.append(loss.item())
            #print epoch and loss
            if verbose:
                print(f"Epoch: {i+1} \tLoss: {round(loss.item(),4)}")
        return history,validationLoss,"epoch"
        
           
    def xavier(self,seed=None):
        """
        Xavier initialization of weights and biases, details not important, code copied from lecture Deep Learning in Scientific Computing.
        """
        if seed==None :
            seed = self.seed
        torch.manual_seed(seed)
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calulate_gain("tanh")
                torch.nn.init.xavier_uniform_(m.weight,gain=g)
                m.bias.data.fill_(0)
            self.apply(init_weights)
            

def showProgress(Tmax=4*np.pi,timesteps=61,initial=[1.,0,0,1],depth=2,width=32,minloss=1e-6,
                 lr=0.25,weights=[1,1,1,0],reltol=1e-7,maxtol=1e12,abstol=1e-7,seed=0):
    """
    Trains the network and shows progress in real time. Plots the phase space coordinates over time. As input takes the final time, the number of intermediate timesteps, the four dimensional initial state, the depth and width of the neural network, the learning rate of the optimizer and the weights in the loss function corresponding to ODE, initial condition, conservation laws, L2 regularization, and supervised learning. Returns a matplotlib animation object, the NeuralNet object, the total loss from training, the State object with the exact solution, and an array with the plotting times.
    """
    t_train = torch.linspace(0,Tmax,timesteps,requires_grad=True,device=device).reshape(-1,1)
    t_plot = torch.linspace(0,Tmax,401,device=device).reshape(-1,1)
    initial = torch.tensor(initial,device=device)
    n = NeuralNet(1,4,depth,width,activation_function=nn.Tanh(),initial=initial,seed=seed)
    n.xavier()
    optimizer = optim.LBFGS(n.parameters(),lr=lr)
    s = State(initial[::2],initial[1::2])
    s.evolve(Tmax/timesteps,Tmax)
    
    fig,ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("Time")
    ax.plot(s.t,s.x,'bx',ms=2,lw=0.7)
    ax.plot(s.t,s.y,'gx',ms=2,lw=0.7)
    ax.plot(s.t,s.px,'rx',ms=2,lw=0.7)
    ax.plot(s.t,s.py,'cx',ms=2,lw=0.7)
    
    qx, = ax.plot([],[],"b-",lw=0.75,label=r"$q_x$")
    qy, = ax.plot([],[],"g-",lw=0.75,label=r"$q_y$")
    px, = ax.plot([],[],"r-",lw=0.75,label=r"$p_x$")
    py, = ax.plot([],[],"c-",lw=0.75,label=r"$p_y$")
    text = ax.text(0.1,-1.05,"")
    plt.legend()
    ax.legend(loc="upper right")

    nsteps = 10
    totalLoss = []
    def animate(i):

        loss,_,_ = n.fit(t_train,nsteps,False,weights,batch_size=None,minloss=minloss,
                       reltol=reltol,abstol=abstol,maxtol=maxtol,optimizer=optimizer)
        totalLoss.append(loss[-1])

        y = n(t_plot).detach().numpy()
        qx.set_data(t_plot.detach().numpy(),y[:,0])
        qy.set_data(t_plot.detach().numpy(),y[:,2])
        px.set_data(t_plot.detach().numpy(),y[:,1])
        py.set_data(t_plot.detach().numpy(),y[:,3])
        text.set_text(f"Loss: {round(1000*loss[-1],4)}\nSteps: {nsteps*(i+1)}\nLearning rate: {round(optimizer.param_groups[0]['lr'],4)}")#"\np: {round(100*p,2)}")#%\nTimesteps: {len(t_train)}")
        return qx,qy,text,px,py,

    an = FuncAnimation(fig,animate,blit=True,interval=0.)
    
    s = State(initial[::2],initial[1::2])
    s.evolve(Tmax/401,Tmax)
    return an,n,totalLoss,s,t_plot

def showProgress2(Tmax=4*np.pi,timesteps=61,initial=[1.,0,0,1],depth=2,width=32,minloss=1e-6,
                  lr=0.25,weights=[1,1,1,0],reltol=1e-7,maxtol=1e12,abstol=1e-7,seed=0):
    """
    Trains the network and shows progress in real time. Plots the trajectory in position space. As input takes the final time, the number of intermediate timesteps, the four dimensional initial state, the depth and width of the neural network, the learning rate of the optimizer and the weights in the loss function corresponding to ODE, initial condition, conservation laws, L2 regularization, and supervised learning. Returns a matplotlib animation object, the NeuralNet object, the total loss from training, the State object with the exact solution, and an array with the plotting times.
    """
    t_train = torch.linspace(0,Tmax,timesteps,requires_grad=True,device=device).reshape(-1,1)
    t_plot = torch.linspace(0,Tmax,401,device=device).reshape(-1,1)
    initial = torch.tensor(initial,device=device)
    n = NeuralNet(1,4,depth,width,activation_function=nn.Tanh(),initial=initial,seed=seed)
    n.xavier()
    optimizer = optim.LBFGS(n.parameters(),lr=lr)
    s = State(initial[::2],initial[1::2])
    s.evolve(Tmax/timesteps,Tmax)
    
    fig,ax = plt.subplots()
    ax.grid()
    ax.set_aspect(1)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.plot([0],[0],'ro',ms=5)
    ax.plot(s.x,s.y,'bx',ms=2,lw=0.7,label="Exact solution")
    pred, = ax.plot([],[],"-",color="orange",lw=1,label=r"$Prediction$")
    text = ax.text(0.1,-1.05,"")
    plt.legend()
    ax.legend(loc="upper right")

    nsteps = 10
    totalLoss = []
    def animate(i):

        loss,_,_ = n.fit(t_train,nsteps,False,weights,batch_size=None,minloss=minloss,
                     reltol=reltol,abstol=abstol,maxtol=maxtol,optimizer=optimizer)
        totalLoss.append(loss[-1])
        
        y = n(t_plot).detach().numpy()
        pred.set_data(y[:,0],y[:,2])
        text.set_text(f"Loss: {round(1000*loss[-1],4)}\nSteps: {nsteps*(i+1)}\nLearning rate: {round(optimizer.param_groups[0]['lr'],4)}")#"\np: {round(100*p,2)}")#%\nTimesteps: {len(t_train)}")
        return pred,text

    an = FuncAnimation(fig,animate,blit=True,interval=0.)
    
    s = State(initial[::2],initial[1::2])
    s.evolve(Tmax/401,Tmax)
    return an,n,totalLoss,s,t_plot

def trainNetwork(Tmax=4*np.pi,timesteps=61,initial=[1.,0,0,1],depth=2,width=32,
                 lr=0.25,weights=[1,1,1,0],epochs=1000,reltol=0,maxtol=1e12,abstol=0-7,minloss=0,seed=2145):
    """
    Trains the network without showing the process. Inputs are the same as showProgress functions but additionally enter the number of training epochs. Returns the NeuralNet, the total loss from the training process, the State object with the exact solution and the time array for plotting.
    """
    t_train = torch.linspace(0,Tmax,timesteps,requires_grad=True,device=device).reshape(-1,1)
    t_plot = torch.linspace(0,Tmax,401,device=device).reshape(-1,1)
    initial = torch.tensor(initial,device=device)
    n = NeuralNet(1,4,depth,width,activation_function=nn.Tanh(),initial=initial,seed=seed)
    n.xavier()
    optimizer = optim.LBFGS(n.parameters(),lr=lr)
    #optimizer = optim.Adam(n.parameters(),lr=lr)
    s = State(initial[::2],initial[1::2])
    s.evolve(Tmax/401,Tmax)
    
    totalLoss,validationLoss,flag = n.fit(t_train,epochs,False,weights,reltol=reltol,minloss=minloss,
                           maxtol=maxtol,abstol=abstol,optimizer=optimizer)
    
    return n,totalLoss,s,t_plot,flag,validationLoss

def showTrainingResult(n,t_plot,totalLoss,s,validationLoss=None):
    """
    Shows result of a training process. As input use the NeuralNet object, the time array for plotting, the total loss from the training process and the State object that contains the exact solution.
    """
    t = t_plot.detach().numpy()
    fig,ax = plt.subplots()
    #ax.set_title("log-scale loss")
    ax.set_xlabel("Epoch")
    if validationLoss is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel("Validation")
        ax.set_ylabel("Training")
        ax2.semilogy()
        ax2.plot(validationLoss,'-',color="orange",label="Validation loss")
        ax2.legend(loc="upper right")
    ax.plot(totalLoss,'b-',label="Training loss")
    ax.semilogy()
    ax.legend(loc="upper left")
    fig,ax = plt.subplots()
    ax.set_aspect(1)
    #plt.title("Trajectory in position space")
    plt.xlabel("x")
    plt.ylabel("y")
    sol = n(t_plot).detach().numpy()
    x,px,y,py = sol[:,0],sol[:,1],sol[:,2],sol[:,3]
    H,L,Ax,Ay = n.conservedQuantitiesNumpy(x,px,y,py)
    plt.plot(x,y,'-',color="blue",label="Prediction")
    plt.plot(s.x,s.y,'--',color="orange",label="Exact solution")
    plt.plot([0],[0],'ro',ms=5)
    #plt.grid()
    plt.legend(loc="upper right")
    plt.figure()
    #plt.title("Predicted phase space coordinates")
    plt.xlabel("Time")
    plt.plot(s.t,s.x,'b--',ms=2)
    plt.plot(s.t,s.y,'g--',ms=2)
    plt.plot(s.t,s.px,'r--',ms=2)
    plt.plot(s.t,s.py,'c--',ms=2)
    plt.plot(t,x,'b-',lw=0.75,label=r"$x$")
    plt.plot(t,y,'g-',lw=0.75,label=r"$y$")
    plt.plot(t,px,'r-',lw=0.75,label=r"$p_x$")
    plt.plot(t,py,'c-',lw=0.75,label=r"$p_y$")
    #plt.grid()
    plt.legend()
    plt.figure()
    #plt.title("Absolute errors")
    plt.xlabel("Time")
    def relativeError(arr):
        return (arr-arr[0])/np.sqrt(arr[0]**2+1e-12)
    def absoluteError(arr):
        return arr-arr[0]
    plt.plot(t,absoluteError(H),lw=0.75,label=r"$H$")
    plt.plot(t,absoluteError(L),lw=0.75,label=r"$L$")
    plt.plot(t,absoluteError(Ax),lw=0.75,label=r"$A_x$")
    plt.plot(t,absoluteError(Ay),lw=0.75,label=r"$A_y$")
    #plt.grid()
    #plt.legend()
    #plt.figure()
    #plt.title("Absolute error of ODE")
    plt.xlabel("Time")
    
    t_plot.requires_grad = True
    u_pred = n(t_plot)
    qx_pred = u_pred[:,0]
    px_pred = u_pred[:,1]
    qy_pred = u_pred[:,2]
    py_pred = u_pred[:,3]

    qx_dot = torch.autograd.grad(qx_pred, t_plot, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(qx_pred))[0]
    px_dot = torch.autograd.grad(px_pred, t_plot, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(px_pred))[0]
    qy_dot = torch.autograd.grad(qy_pred, t_plot, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(qy_pred))[0]
    py_dot = torch.autograd.grad(py_pred, t_plot, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(py_pred))[0]

    qx_dot = qx_dot.flatten()
    px_dot = px_dot.flatten()
    qy_dot = qy_dot.flatten()
    py_dot = py_dot.flatten()
    qx_pred = qx_pred.flatten()
    px_pred = px_pred.flatten()
    qy_pred = qy_pred.flatten()
    py_pred = py_pred.flatten()
    t_plot = t_plot.flatten()
    
    t_plot = t_plot.detach()
    
    d3 = (qx_pred.square()+qy_pred.square()).pow(1.5)
    plt.plot(t_plot,(qx_dot-px_pred).detach(),'--',lw=0.75,label=r"$\dot{x}-p_x$")
    plt.plot(t_plot,(qy_dot-py_pred).detach(),'--',lw=0.75,label=r"$\dot{y}-p_y$")
    plt.plot(t_plot,(d3*px_dot+qx_pred).detach(),'--',lw=0.75,label=r"$q^3\dot{p}_x+x$")
    plt.plot(t_plot,(d3*py_dot+qy_pred).detach(),'--',lw=0.75,label=r"$q^3\dot{p}_y+y$")
    plt.legend()
    
def L2Error(xexact,yexact,xpred,ypred):
    try:
        return np.mean((xexact-xpred)**2+(yexact-ypred)**2)
    except:
        return np.mean((xexact-xpred)**2+(yexact-ypred)**2)