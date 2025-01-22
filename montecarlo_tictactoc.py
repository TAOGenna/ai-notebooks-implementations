import numpy as np
import random
import copy 

class Board: 
    def __init__(self):
        self.board = np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0]],dtype=int)
    
    def UntriedMove(self, turn):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i,j))
        return random.choice(moves) 

    def IsTerminal(self):
        # if there is no hope for a winner or if there is a winner
        
        # analyse if there is winner 
        for i in range(3): 
            if np.absolute(np.sum(self.board[i,:]))==3 or np.absolute(np.sum(self.board[:,i]))==3: return True
        if np.absolute(np.sum([self.board[i,i] for i in range(3)]))==3 or np.absolute(np.sum([self.board[i,2-i] for i in range(3)]))==3: return True

        # analyse if there is a draw | the board is completely filled
        if np.sum(self.board == 0) == 0: return True

        # return answer 
        return False 

    def Winner(self):
        # Ask if it is a terminal state first| returns -1(AI), +1(human), 0(draw)

        # analyse if there is winner 
        for i in range(3): 
            if np.sum(self.board[i,:])==3 or np.sum(self.board[:,i])==3: return 1
            if np.sum(self.board[i,:])==-3 or np.sum(self.board[:,i])==-3: return -1

        if np.sum([self.board[i,i] for i in range(3)])==3 or np.sum([self.board[i,2-i] for i in range(3)])==3: return 1
        if np.sum([self.board[i,i] for i in range(3)])==-3 or np.sum([self.board[i,2-i] for i in range(3)])==-3: return -1

        return 0 # draw 
    
    def LegalMoves(self):
        if self.IsTerminal(): return 0
        return np.sum(self.board == 0)
    

class Node:
    def __init__(self, state, parent = None):
        self.state = state
        # Node properties 
        self.Q = 0.0 # Q(v): total reward of all playouts that passed through this state
        self.N = 1 # N(v): number of times it has been visited
        # Graph stuff
        self.children  = [] #  of nodes
        self.parent  = parent
    
    def AddNode(self, child_state):
        new_node = Node(state=child_state, parent=self)
        self.children.append(new_node)
    
    def FullyExplored(self):
        if len(self.children) == self.state.LegalMoves():
            return True
        return False
    
    def UCB(self,cte,visits_parent):
        return self.Q/self.N + cte*np.sqrt(2*np.log(visits_parent)/self.N)

def MCTS(budget, root):

    for _ in range(budget):
        leaf, turn = TreePolicy(node=root, turn=-1)
        #print(f'number of childs = {len(root.children)}')
        reward = DefaultPolicy(leaf.state, turn)
        BackupNegamax(leaf, reward, turn)

    print(np.array([children.UCB(0,root.N) for children in root.children]))
    return BestChild(root, 0)

def TreePolicy(node, turn) -> Node:
    Cp = 1.0/np.sqrt(2) 

    while node.state.IsTerminal() == False and node.state.Winner() == 0:

        if node.FullyExplored() == False: 
            return Expand(node, turn), -turn 
        else:
            node = BestChild(node, Cp)
            turn = -turn
    return node, turn

def BestChild(node, c) -> Node:
    best_UCB = np.array([children.UCB(c,node.N) for children in node.children])
    best_value = np.max(best_UCB)
    rand_chosen = []
    for idx,ucb in enumerate(best_UCB):
        if ucb==best_value:
            rand_chosen.append(idx)
    return  node.children[random.choice(rand_chosen)]

def Expand(node, turn) -> Node:
    action = node.state.UntriedMove(turn)
    new_state = copy.deepcopy(node.state)
    new_state.board[action[0]][action[1]] = turn
    node.AddNode(child_state=new_state)
    return node.children[-1]

def DefaultPolicy(state, turn) -> int:
    state = copy.deepcopy(state)
    while state.IsTerminal() is False and state.Winner() == 0:  
        action = state.UntriedMove(turn)
        state.board[action[0]][action[1]] = turn
        turn = -turn
    return state.Winner()

def BackupNegamax(node, reward, turn):
    cnt = 0 
    while node is not None:
        cnt += 1
        #print(node.state.board)
        node.N += 1
        node.Q -= turn*reward
        turn = -turn
        node = node.parent
    #print(f'number of layers = {cnt}')
    #print('------------------------------------------------')
    return 
    


#--------------------------------------------------------------------------------------------------
#------------------------- MONTE CARLO TREE SEARCH  - TIC TAC TOE ---------------------------------
#--------------------------------------------------------------------------------------------------

def check_game(state):
    if state.IsTerminal() and state.Winner():
        print('------------------ GAME FINISHED ---------------------------')
        if state.Winner() == 0:
            print('DRAW')
        elif state.Winner() == 1:
            print('HUMAN')
        else:
            print('AI')
        print('Final Board \n',state.board)
        return True
    return False


ITERATIONS = 1000
state = Board()
while True:
    print('Current state of the game: \n', state.board)
    
    # user enters a response
    cellx, celly = map(int, input("Enter two numbers separated by space: ").split())
    cellx -=1
    celly -=1
    state.board[cellx][celly] = 1
    print('board after my action \n',state.board)

    if check_game(state): break
    
    best_action = MCTS(ITERATIONS, Node(state=copy.deepcopy(state)))
    state = copy.deepcopy(best_action.state)

    if check_game(state): break
