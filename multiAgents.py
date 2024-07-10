# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print(successorGameState)
        #print(newPos)
        #print(newFood.asList())
        
        #print(newScaredTimes)
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        foodDistances.append(10000)
        closestFoodPosDist = min(foodDistances)
        closestGhostDist = min([manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])
        ghostFactor = 0
        foodFactor = 10*(20-closestFoodPosDist)
        if closestGhostDist <= 1:
            ghostFactor = 500
        return successorGameState.getScore()+ 1/closestFoodPosDist - ghostFactor

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, gameState: GameState, depth, agent):
        if (depth == 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        #if pacman (maximizing player)
        if agent == 0:
            value = -1000000
            for direction in gameState.getLegalActions(agent):
                successorState = gameState.generateSuccessor(agent, direction)
                value = max(value, self.minimax(successorState, depth, agent + 1))     
            return value
        else:
            value = 1000000
            for direction in gameState.getLegalActions(agent):
                successorState = gameState.generateSuccessor(agent, direction)
                if (agent == (gameState.getNumAgents() - 1)):
                    value = min(value, self.minimax(successorState, depth - 1, 0))
                else:
                    value = min(value, self.minimax(successorState, depth, agent + 1))
            return value
        
    
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        bestAction = None
        bestActionValue = -100000
        for action in gameState.getLegalActions(0):
            actionValue = 0
            successorState = gameState.generateSuccessor(0, action)
            actionValue = self.minimax(successorState, self.depth, 1)
            if actionValue > bestActionValue:
                bestActionValue = actionValue
                bestAction = action
                
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, gameState: GameState, depth, agent, alpha, beta):
        if (depth == 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        #if pacman (maximizing player)
        if agent == 0:
            value = -1000000
            for direction in gameState.getLegalActions(agent):
                successorState = gameState.generateSuccessor(agent, direction)
                value = max(value, self.alphabeta(successorState, depth, agent + 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        else:
            value = 1000000
            for direction in gameState.getLegalActions(agent):
                successorState = gameState.generateSuccessor(agent, direction)
                if (agent == (gameState.getNumAgents() - 1)):
                    value = min(value, self.alphabeta(successorState, depth - 1, 0, alpha, beta))
                else:
                    value = min(value, self.alphabeta(successorState, depth, agent + 1, alpha, beta))
                if (value < alpha):
                    return value
                beta = min(beta, value)
            return value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction = None
        bestActionValue = -100000
        alpha = -10000000
        beta = 10000000
        for action in gameState.getLegalActions(0):
            actionValue = 0
            successorState = gameState.generateSuccessor(0, action)
            # Wrong alpha beta values, change before running
            actionValue = self.alphabeta(successorState, self.depth, 1, alpha, beta)
            if actionValue > bestActionValue:
                bestActionValue = actionValue
                bestAction = action
            alpha = max(alpha, actionValue)
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState: GameState, depth, agent):
        if (depth == 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        #if pacman (maximizing player)
        if agent == 0:
            value = -1000000
            for direction in gameState.getLegalActions(agent):
                successorState = gameState.generateSuccessor(agent, direction)
                value = max(value, self.expectimax(successorState, depth, agent + 1))     
            return value
        else:
            value = 0
            legalActions = gameState.getLegalActions(agent)
            i = 0
            for direction in legalActions:
                i += 1
                successorState = gameState.generateSuccessor(agent, direction)
                if (agent == (gameState.getNumAgents() - 1)):
                    value +=  self.expectimax(successorState, depth - 1, 0)
                else:
                    value +=  self.expectimax(successorState, depth, agent + 1)
            return value / i
        
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        bestAction = None
        bestActionValue = -100000
        for action in gameState.getLegalActions(0):
            actionValue = 0
            successorState = gameState.generateSuccessor(0, action)
            actionValue = self.expectimax(successorState, self.depth, 1)
            if actionValue > bestActionValue:
                bestActionValue = actionValue
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Rates the games tate by its current score, and incentivizes getting closer to food
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newCapsules = currentGameState.getCapsules( )
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    foodDistances.append(10000)
    closestFoodPosDist = min(foodDistances)
    closestGhostDist = min([manhattanDistance(newPos, ghostPos) for ghostPos in currentGameState.getGhostPositions()])
    ghostFactor = 0
    foodFactor = 10*(20-closestFoodPosDist)
    if closestGhostDist <= 1:
        ghostFactor = 500
    return 5 * currentGameState.getScore()+ 1/closestFoodPosDist  - ghostFactor
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
