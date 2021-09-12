
def evaluationFunction(self, currentGameState, action):

  """

  Design a better evaluation function here.



  The evaluation function takes in the current and proposed successor

  GameStates (pacman.py) and returns a number, where higher numbers are better.



  The code below extracts some useful information from the state, like the

  remaining food (oldFood) and Pacman position after moving (newPos).

  newScaredTimes holds the number of moves that each ghost will remain

  scared because of Pacman having eaten a power pellet.



  Print out these variables to see what you're getting, then combine them

  to create a masterful evaluation function.

  """

  #Useful information you can extract from a GameState (pacman.py)
  successorGameState = currentGameState.generatePacmanSuccessor(action)
  newPos = successorGameState.getPacmanPosition()
  oldFood = currentGameState.getFood()
  newFood = successorGameState.getFood()
  newFoodList = newFood.asList()
  ghostPositions = successorGameState.getGhostPositions()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  print('Successor game state:\n', successorGameState)
  print('Pacman current position: ', newPos)
  print('oldFood:\n', oldFood)
  print('newFood:\n', newFood)
  print('ghostPositions: ', ghostPositions)
  print('successorGameState.score: ', successorGameState.getScore())
  print('newScaredTimes: ', newScaredTimes)


  # computa distância para o fantasma mais próximo.

  minDistanceGhost = float("+inf")

  for ghostPos in ghostPositions:
    minDistanceGhost = min(minDistanceGhost, util.manhattanDistance(newPos, ghostPos))



  # se a acao selecionada leva à colisão com o ghost, pontuação é mínima

  if minDistanceGhost == 0:

    return float("-inf")

  

  # se a acao conduzir para a vitoria, pontuação é máxima

  if successorGameState.isWin():

    return float("+inf")

  

  score = successorGameState.getScore()



  # incentiva acao que conduz o agente para mais longe do fantasma mais próximo

  score += 2 * minDistanceGhost

  

  minDistanceFood = float("+inf")

  for foodPos in newFoodList:

    minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

  

  # incentiva acao que conduz o agente para mais perto da comida mais próxima

  score -= 2 * minDistanceFood

  

  # incentiva acao que leva a uma comida

  if(successorGameState.getNumFood() < currentGameState.getNumFood()):

    score += 5

  

  # penaliza as acoes de parada

  if action == Directions.STOP:

    score -= 10

  

  return score