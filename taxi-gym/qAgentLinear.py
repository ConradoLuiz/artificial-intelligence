import numpy as np

class QLearningAgentLinear:
  
  def __init__(self, 
               n_states, 
               n_actions,
               env, 
               decay_rate = 0.0001, 
               learning_rate = 0.7, 
               gamma = 0.618):
    
    self.env = env
    self.n_actions = n_actions
    self.n_states = n_states
    # self.q_table = np.zeros((n_states, n_actions))
    self.epsilon = 1.0
    self.max_epsilon = 1.0
    self.min_epsilon = 0.01
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.gamma = gamma # discount rate
    self.epsilons_ = []

    self.n_features = 2
    self.weights = [1 for _ in range(self.n_features)]

    self.memoization = {i: {} for i in range (self.n_features)}

    
  def choose_action(self, state=None, explore=True):
    exploration_tradeoff = np.random.uniform(0, 1)
    
    if explore and exploration_tradeoff < self.epsilon:
      # exploration
      return np.random.randint(self.n_actions)    
    else:
      # exploitation (taking the biggest Q value for this state)
      _state = state if state is not None else self._state
      return np.argmax([self.Q(_state, action) for action in range(self.n_actions)])

  def Q(self, state, action):
    value = 0

    # for i in range(self.weights.size):
    #   function = getattr(self, f'f{i}')
    #   value += self.weights[i] * function(state, action)

    value += self.weights[0] * self.f0(state, action)
    value += self.weights[1] * self.f1(state, action)
    return value
    
  def learn(self, state, action, reward, next_state, done, episode):
    self._state = state

    diff = ( reward + self.gamma * np.max([self.Q(next_state, a) for a in range(self.n_actions)]) ) - self.Q(state, action)

    # Update the weights
    # for i in range(self.weights.size):
    #   function = getattr(self, f'f{i}')
    #   self.weights[i] = self.weights[i] + self.learning_rate * diff * function(state, action)

    self.weights[0] = self.weights[0] + self.learning_rate * diff * self.f0(state, action)
    self.weights[1] = self.weights[1] + self.learning_rate * diff * self.f1(state, action)

    if done:
      # Reduce epsilon to decrease the exploration over time
      self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
        np.exp(-self.decay_rate * episode)
      self.epsilons_.append(self.epsilon)

  def indexToPos(self, index):
    return {
        '0': (0,0),
        '1': (0,4),
        '2': (4,0),
        '3': (4,3)
    }[str(index)]

  def distance(self, p1, p2):
    return np.sqrt( ((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2) )

  def f0(self, state, action):
    '''
    Inverso da distância do agente até o ponto de desembarque quando o carro está ocupado 
    (retorna 0 quando o carro está desocupado)
    '''
    if state in self.memoization[0]:
        return self.memoization[0][state]
    
    row_taxi, col_taxi, passenger, destination = self.env.unwrapped.decode(state)
    
    if passenger <= 3:
      return 0 

    d = self.distance(self.indexToPos(destination), (row_taxi, col_taxi))

    return_value = 1/d if d != 0 else 2

    self.memoization[0][state] = return_value

    return return_value

  def f1(self, state, action):
    '''
    Inverso da distância do agente até o ponto de embarque quando o carro está vazio 
    (retorna 0 quando o carro está ocupado)
    '''
    if state in self.memoization[1]:
        return self.memoization[1][state]

    row_taxi, col_taxi, passenger, destination = self.env.unwrapped.decode(state)
    
    if passenger > 3:
      return 0 

    d = self.distance((row_taxi, col_taxi), self.indexToPos(passenger))

    return_value = 1/d if d != 0 else 2

    self.memoization[0][state] = return_value

    return return_value