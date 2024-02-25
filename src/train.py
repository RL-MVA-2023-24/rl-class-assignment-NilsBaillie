from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import numpy as np
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def collect_samples(horizon):
        s,_ = env.reset()
        S,A,R,S2 = [],[],[],[]
        for _ in range(horizon):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            if trunc:
                s, _ = env.reset()
            else :
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        return S, A, R, S2 

def FQI(S,A,R,S2,n_iter,nb_actions,gamma):
    Qfuncs = []
    nb_samples = S.shape[0]
    SA = np.append(S,A,axis=1)
    for it in range(n_iter):
        if it == 0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((nb_samples,1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfuncs[-1].predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*max_Q2
        #Q = RandomForestRegressor()
        Q = ExtraTreesRegressor(n_estimators=15)
        Q.fit(SA,value)
        Qfuncs.append(Q)
    return Qfuncs

class ProjectAgent:
    def __init__(self, Qfunction = None):
        self.n_actions = env.action_space.n
        self.Qfunction = Qfunction
        
    def act(self, observation, use_random=False):
        if use_random :
            return np.random.randint(0, self.n_actions - 1)
        else :
            Qvals = []
            for a in range(self.n_actions):
                obs_a = np.append(observation,a).reshape(1,-1)
                Qvals.append(self.Qfunction.predict(obs_a))
            return np.argmax(Qvals)

    def save(self, path = 'FQI_ExtraTrees.pkl'):
        joblib.dump(self.Qfunction, path)

    def load(self):
        self.Qfunction = joblib.load('FQI_ExtraTrees.pkl')

# Train and save
#np.random.seed(1)
#horizon = 10**4
#nb_actions = env.action_space.n
#S, A, R, S2 = collect_samples(horizon)
#n_iter = 50
#gamma = 0.95
#Qfuncs = FQI(S, A, R, S2, n_iter, nb_actions, gamma)
#Qfunction = Qfuncs[-1]
#agent = ProjectAgent(Qfunction)
#agent.save()

