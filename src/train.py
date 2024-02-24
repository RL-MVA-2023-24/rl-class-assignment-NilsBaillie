from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, env=env, n_iter=10**3, horiz = 200, gamma=0.9, epsilon=0.1):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.Qfct = RandomForestRegressor()
        self.horiz = horiz
        self.n_iter = n_iter
    
    #def epsilon_greedy(self, s):
    #    if np.random.random() < self.epsilon:
    #        return np.random.randint(0, self.n_actions - 1)
    #    else:
    #        Qvals = self.Qfct.predict([s])
    #        return np.argmax(Qvals)
#
    #def get_data(self,greedy):
    #    s1 = self.env.reset()
    #    for i in range(self.batch_size):
    #        if greedy :
    #            a = self.epsilon_greedy(s1)
    #        else :
    #            a = np.random.randint(0, self.n_actions - 1)
    #        s2, r,_,_,_ = self.env.step(a)
    #        self.dataset.append((s1, a, r, s2))
    #        s1 = s2
    
    def collect_samples(self):
        s,_ = self.env.reset()
        S,A,R,S2 = [],[],[],[]
        for _ in range(self.horiz):
            a = env.action_space.sample()
            s2, r,_,_, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        return S, A, R, S2      
    
    def FQI(self):
        S, A, R, S2 = self.collect_samples()
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        for it in range(self.n_iter):
            if it == 0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,self.n_actions))
                for a2 in range(self.n_actions):
                    A2 = a2*np.ones((nb_samples,1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Qfct.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + self.gamma*max_Q2
            self.Qfct = RandomForestRegressor()
            self.Qfct.fit(SA,value)  
        
    def act(self, observation, use_random=False):
        if use_random :
            return np.random.randint(0, self.n_actions - 1)
        else :
            Qvals = []
            for a in range(self.n_actions):
                obs_a = np.append(observation,a).reshape(1,-1)
                Qvals.append(self.Qfct.predict(obs_a))
            return np.argmax(Qvals)


    def save(self, path = 'FQI_agent.pkl'):
        joblib.dump(self.Qfct, path)

    def load(self):
        self.Qfct = joblib.load('FQI_agent.pkl')
