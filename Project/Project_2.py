#!/usr/bin/env python
# coding: utf-8

# The goal of this program is to develop a simple model to capture the dynamics of a virus infecting an algal host. 

# # Preliminary Data

# The data we are going to use is from a 2005 paper by Baudoux and Brussaard ('Characterization of different viruses infecting the marine harmful algal bloom species Phaeocystis globosa')
# 
# First, let's graph the control data. We can first read in the data

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.integrate import odeint


# In[7]:

control_df = pd.read_csv('/Users/francescaravelli/Desktop/Project/Data/phaeocystis_control.csv')

# The 'treatment' data are stored in the file '../data/phaeocystis_PgV_one_step.csv'. 

# In[8]:


treat_df = pd.read_csv('/Users/francescaravelli/Desktop/Project/Data/phaeocystis_PgV_one_step.csv')


# In[9]:


control_text = control_df.to_string()
treat_text = treat_df.to_string()

# # Comparing results of number of Infection classes

# Making a function that does the euler integration for given n infection classes. Having some trouble with it and I haven't gotten it to work yet.

# In[66]:


def inf_class(u ,t, tau, mu, phi, beta, n):
        
    S = u[0]
    I = u[1:-1]
    V = u[-1]

    dIsdt = np.zeros(n)
    dIsdt[0] = phi * S * V - (I[0] / tau) 
        
    for i in range(1, n): 
        dIsdt[i] = (I[i-1] / tau) - (I[i] / tau)
        
    dVdt = beta * (I[-1] / tau) - phi * S * V
    dSdt = mu * S - phi * S * V

    return np.concatenate((np.r_[[dSdt]],dIsdt,np.r_[[dVdt]]))

# define parameters
def params(S, V, n):
    
    tau=0.45
    mu=1
    phi=1.8e-7
    beta=1000
    
    n = n
    
    Is = np.zeros(n)
    inits = np.concatenate((np.r_[[S]],Is,np.r_[[V]]))
    
    return (tau,mu,phi,beta,n), Is, inits


# In[24]:
 # Plot results
pl.figure()
for i in range(1,6):
    n=i
    params_vals, Is, inits = params(1420000, 25000000, n)

    # define time domain
    tmin=0
    tmax=3
    delta=0.01
    nsteps = int((tmax - tmin) / delta)
    times = np.linspace(tmin, tmax, nsteps)
    
    
    sols = odeint(inf_class,inits,times,args=params_vals).T
    
    # Plot results
    
    pl.semilogy(times,sum(sols[0:-1]),label=f'I{i}')
    
pl.scatter(treat_df['time'],treat_df['host'])
pl.xlabel('Time')
pl.ylabel('Population')

pl.show()
     
pl.figure()      
for i in range(1,6):
    n=i
    
    params_vals, Is, inits = params(1420000, 25000000, n)

    # define time domain
    tmin=0
    tmax=3
    delta=0.01
    nsteps = int((tmax - tmin) / delta)
    times = np.linspace(tmin, tmax, nsteps)
    
    sols = odeint(inf_class,inits,times,args=params_vals).T
   
    pl.semilogy(times,sols[-1],label=f'I{i}')

pl.scatter(treat_df['time'],treat_df['virus'])
pl.xlabel('Time')
pl.ylabel('Virus (ml$^{-1}$)')

pl.show()




