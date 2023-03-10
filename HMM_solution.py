import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Y = Observations
        self.T = Transition
        self.M = Emission
        self.pi = Initial_distribution

    def forward(self):
        alpha = np.zeros((self.Y.shape[0], self.T.shape[0]))
        # M = M_xy
        alpha[0,:] = self.pi * self.M[:, self.Y[0]]
        for i in range(1,self.Y.shape[0]):
            alpha[i,:] = alpha[i-1,:] @ self.T * self.M[:,self.Y[i]]

        return alpha

    def backward(self):
        beta_in  = 1 
        beta = np.zeros((self.Y.shape[0], self.T.shape[0]))
        for j in range(self.Y.shape[0]):
            beta[j,:] = np.dot(np.multiply(beta_in, self.M[:, self.Y[j]]), self.T.T)
            
        return beta

    def gamma_comp(self, alpha, beta):
        gamma = np.zeros((self.Y.shape[0], self.T.shape[0]))
        for i in range(self.Y.shape[0]):
            gamma[i,:] = np.multiply(alpha[i,:], beta[i,:])
            gamma[i,:] /= np.sum(gamma[i,:])

        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.zeros((self.Y.shape[0]-1, self.T.shape[0], self.T.shape[0]))
        for k in range(self.Y.shape[0]-1):
            a = self.Y[k+1]
            ch1= np.dot(alpha[k,:],self.T)
            ch2 = np.multiply(self.M[:,a],beta[k+1,:])
            ch3 = np.outer(ch1,ch2)
            ch4 = np.divide( ch3, np.sum(ch3))
            xi[k,:] = ch4
           
        return xi

    def update(self, alpha, beta, gamma, xi):
        M_prime = np.zeros_like(self.M)
        T_prime = np.zeros_like(self.T)
        new_init_state = np.zeros_like(self.pi)
        new_init_state = gamma[0,:] 
        T_prime = np.divide(np.sum(xi, axis=0), np.sum(gamma, axis=0))
        Num = np.zeros((self.Y.shape[0], 3))
        for k in range(self.Y.shape[0]):
            for l in range(3):
                if l==self.Y[k]:
                    Num[k,l] = 1
        M_prime = np.divide(gamma.T @ Num, np.sum(gamma, axis=0).reshape(-1,1))

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        P_original = np.sum(alpha[-1,:])
        self.T = T_prime
        self.M = M_prime
        self.pi = new_init_state
        P_prime = np.sum(self.forward()[-1,:])
      
        return P_original, P_prime
