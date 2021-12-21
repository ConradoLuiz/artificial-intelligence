import numpy as np

class QLearningAgentTabular:

    def __init__(
        self,
        n_states,
        n_action,
        decay_rate = .0001,
        learning_rate = .7,
        gamma = 0.618    
    ):

        self.n_actions = n_action
        self.q_table = np.zeros((n_states, n_action))
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = .01
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilons_ = []
    
    def choose_action(self, explore=True):
        exploration_tradeoff = np.random.uniform(0, 1)

        if explore and exploration_tradeoff < self.epsilon:
            return np.random.randint(self.n_actions)
        
        else:
            return np.argmax(self.q_table[self._state, :])
    
    def learn(self, state, action, reward, next_state, done, episode):
        self._state = state

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        self.q_table[state, action] = self.q_table[state, action] \
                                    + self.learning_rate \
                                    * \
                                    ( 
                                        reward + self.gamma \
                                        * np.max( self.q_table[next_state, :] - self.q_table[state, action] ) 
                                    )

        if done:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            self.epsilons_.append(self.epsilon)
