import numpy as np

def wiener(x0,n_steps,T,gamma,sigma):
    dt = T/n_steps
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps+1)
    x[0] = x0
    for i in range(n_steps):
        x[i+1] = x[i] + gamma*dt + sigma*np.sqrt(dt)*np.random.normal()
    return t, x

def geo_brownian(S0, n_steps,T, mu, sigma):
    W= wiener(0,n_steps,T,0,1)[1]   
    t = np.linspace(0, T, n_steps + 1)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t,S

def brownian_bridge(x0,x1,n_steps,T):
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps+1)
    W = wiener(x0,n_steps,T,0,1)[1]
    x[0] = x0
    x[n_steps] = x1
    for i in range(1,n_steps):
        x[i] = W[i] - (t[i]/T)*(W[-1]-x1)    
    return t, x

def max_process(x0,n_steps,T,gamma,sigma):
    W = wiener(x0,n_steps,T,gamma,sigma)[1]
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps+1)
    x[0] = x0
    for i in range(1,n_steps+1):
        x[i] = max(x[i-1],W[i])
    return t, x

