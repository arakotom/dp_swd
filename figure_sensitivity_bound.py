#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


from scipy.stats import beta
import numpy as np
import  matplotlib.pyplot as plt

d = 784
num_projections = 200  
sigma = 1
n_bin = 100
C = 0.5
#-----------------------------------------------------------------------------
#                       creating x, xprime and z
#-----------------------------------------------------------------------------
x = np.random.randn(d)*sigma
xp = np.random.randn(d)*sigma
z = (x-xp)/np.linalg.norm(x-xp)
N = 10000
liste = []
liste_single =[]
for i in range(N):
    projections = np.random.randn(d,num_projections)*sigma    
    X = (z@projections )/ np.sqrt(np.sum(projections ** 2, axis=0))
    liste.append(np.sum(X**2))
    # looking at a single projection
    u = projections[:,0]/np.sqrt(np.sum(projections[:,0]**2))
    xs= np.sum((x-xp)/np.linalg.norm(x-xp)*u)**2
    liste_single.append(xs)


#%%
vec, bine = np.histogram(liste,n_bin)
vec = vec/np.sum(vec)
vec_s, bine_s = np.histogram(liste_single,n_bin)
vec_s = vec_s/np.sum(vec_s)




print('Estimated expectation :', np.mean(liste),'True expectation :', num_projections/d)
print('Estimated variance :', np.var(liste), 'True Variance :', np.var(liste_single)*num_projections)
a = 1/2
b = (d - 1)/2 
v_est = a*b/((a+b)**2*(a+b+1))*num_projections
delta = 1e-5
print('upper_bound at 5sigma =',num_projections/d+5*np.sqrt(v_est)) 
print('upper_bound at 10sigma of Sqrt(Y) =', np.sqrt(num_projections/d+10*np.sqrt(v_est))) 

delta = 1e-5
print(beta.isf(delta,a,b))


#%%
maxi = 4
from scipy.stats import norm
# computing 1 -cdf(delta)
delta = 1e-5
d5 = norm.isf(delta,num_projections/d,np.sqrt(v_est))

delta = 1e-6
d6 = norm.isf(delta,num_projections/d,np.sqrt(v_est))

# showing mean + 5standdev
s6 = num_projections/d+6*np.sqrt(v_est)

zd = norm.isf(1e-5,0,1)
clt_b =  num_projections/d + zd/d* np.sqrt((2*num_projections*(d-1))/(d+2)) 


plt.plot(bine[1:],vec*n_bin*5,label='Estimation',linewidth=2)
plt.plot(bine[1:],norm.pdf(bine[1:],num_projections/d,np.sqrt(v_est)),label='True Distribution',linewidth=2)
plt.vlines(num_projections/d, 0, maxi, colors='r', label= 'mean',linestyles='--',linewidth=2)
plt.vlines(d5, 0, maxi,colors='k',linestyles='--', label='x : 1 - cdf(x) = 1e-5',linewidth=2)
plt.vlines(d6, 0, maxi,colors='g',linestyles='--', label='x : 1 - cdf(x) = 1e-6',linewidth=2)
plt.vlines(s6, 0, maxi,colors='m',linestyles='--', label = 'm + 6s',linewidth=2)
# CLT bound coincides with d5 and d6 bound as we are a good CLT approximation
#plt.vlines(clt_b, 0, maxi,colors='c',linestyles='--', label = 'CLT Bound')

plt.grid()
plt.legend()
plt.xlabel('x', fontsize=14)
plt.ylabel('estimated histogram at x', fontsize=14)
plt.ylim([0,20])
filename = f"NormalApproximation.png"
plt.show()

#plt.savefig(filename,dpi=300,bbox_inches='tight')


