import numpy as np
import matplotlib.pyplot as plt

def compute_Qmu(alpha, J_mu, mdp):
    '''
    Returns Qmu, an mdp.nS x mdp.nA array representing the Q function at
    cost vector J 
    '''
    Qmu = np.array([[sum([l[0]*(l[2] + alpha*J_mu[l[1]]) for l in mdp.P[s][a]]) for a in range(mdp.nA)] for s in range(mdp.nS)])    
    return Qmu

def compute_Jmu(alpha, mdp, mu):
    '''
    Solve system (I-alpha*P_mu)J = g_mu, 
    where J is cost under mu, P_mu is transition matrix, and g_mu is cost per stage
    '''
    #Get one-stage costs and transition matrix under policy mu
    g_mu = np.array([sum([l[0]*l[2] for l in mdp.P[s][mu[s]]]) for s in mdp.transient])
    P_mu = np.zeros((mdp.nS, mdp.nS))
    for s in range(mdp.nS):
        for l in mdp.P[s][mu[s]]:
            P_mu[s,l[1]] += l[0]            
    P_mu = P_mu[mdp.transient,:]    
    P_mu = P_mu[:,mdp.transient]
    
    #Solve Bellman equation at transient states
    J = np.zeros(mdp.nS)
    J[mdp.transient] = np.linalg.solve(np.identity(mdp.nS-5)-alpha*P_mu, g_mu)    
    return J

def policy_iteration(mdp, alpha, nIt):
    mu_k = np.zeros(mdp.nS,dtype='int') #Initialize policy
    mus = [mu_k]   #Lists to collect iterates
    Js = []
    for it in range(nIt):   
        J_mu = compute_Jmu(alpha, mdp, mu_k)  #Value step
        Q_mu = compute_Qmu(alpha, J_mu, mdp)             #Policy step
        mu_kp1 = Q_mu.argmax(axis=1)
        Js.append(J_mu)
        mus.append(mu_kp1)
        mu_k = mu_kp1
    return Js, mus

def plot_policy(env, Js, mus, nIt=7):
  for (J, mu) in zip(Js[:nIt], mus[:nIt]):
      plt.figure(figsize=(3,3))
      plt.imshow(J.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
      ax = plt.gca()
      ax.set_xticks(np.arange(4)-.5)
      ax.set_yticks(np.arange(4)-.5)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      Y, X = np.mgrid[0:4, 0:4]
      a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
      Mu = mu.reshape(4,4)
      for y in range(4):
          for x in range(4):
              a = Mu[y, x]
              u, v = a2uv[a]
              plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
              plt.text(x, y, str(env.desc[y,x].item().decode()),
                       color='g', size=12,  verticalalignment='center',
                       horizontalalignment='center', fontweight='bold')
      plt.grid(color='b', lw=2, ls='-')