import numpy as np

__all__ = ["EM_hopf"]

def EM_hopf(M=100, N=2**16, xlambda=-1., omega=1, sigma=.5,
            Xzero=1, Yzero=0, T=10):
    """
    EM  Stochastic Hopf bifurcation

    Discretized Brownian path over [0,T] has dt = 2^(-8).
    Euler-Maruyama uses timestep R*dt.
    """
    np.random.seed(M)

    # xlambda = -1.0; omega = 1; sigma = 0.5
    # Xzero = 1; Yzero = 0
    # T=10; N=2**16;
    dt = float(T)/N
    t=np.linspace(0,T,N+1)

    dW1=np.sqrt(dt)*np.random.randn(1,N)
    W1=np.cumsum(dW1)
    dW2=np.sqrt(dt)*np.random.randn(1,N)
    W2=np.cumsum(dW1)

    R=1; Dt=R*dt; L=float(N)/R
    Xem=np.zeros(L+1); Xem[0] = Xzero
    Yem=np.zeros(L+1); Yem[0] = Yzero

    for j in range(1,int(L)+1):
        Winc1=np.sum(dW1[0][range(R*(j-1),R*j)])
        Winc2=np.sum(dW2[0][range(R*(j-1),R*j)])
        Rtemp = Xem[j-1]*Xem[j-1] + Yem[j-1]*Yem[j-1]
        Xdrift = xlambda*Xem[j-1]-omega*Yem[j-1]-Xem[j-1]*Rtemp
        Ydrift = xlambda*Yem[j-1]+omega*Xem[j-1]-Yem[j-1]*Rtemp
        Xem[j] = Xem[j-1] + Dt*Xdrift + sigma*Winc1
        Yem[j] = Yem[j-1] + Dt*Ydrift + sigma*Winc2

    return Xem, Yem
