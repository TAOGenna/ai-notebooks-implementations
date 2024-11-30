import numpy as np
import random
import copy 
from tqdm import tqdm

class Board: 
    def __init__(self):
        self.row = 6
        self.col = 7
        self.board = np.zeros(shape=(self.row,self.col),dtype=int)
        self.next = np.zeros(self.col,dtype=int) # to indicate the available height cell of the column
    
    def UntriedMove(self):
        moves = []
        for index,available_height in enumerate(self.next):
            if available_height<6: moves.append((available_height,index))
        choice = random.choice(moves)
        return choice 

    def IsTerminal(self):
        if self.Winner() != 0:
            return True
        # the board is completely filled
        if np.sum(self.board == 0) == 0: return True 
        return False 

    def Winner(self):# Ask if it is a terminal state first| returns -1(AI), +1(human), 0(draw)
        def check_line(line):
            for i in range(len(line)-3):
                if line[i] != 0 and line[i] == line[i+1] == line[i+2] == line[i+3]: return line[i] 
            return 0

        for i in range(self.row): 
            if (val := check_line(self.board[i,:])): return val
        for i in range(self.col): 
            if (val := check_line(self.board[:,i])): return val 
        # Check diagonals (bottom-left to top-right)
        for row in range(self.row - 3):
            for col in range(self.col - 3):
                if self.board[row, col] != 0 and self.board[row, col] == self.board[row+1, col+1] == self.board[row+2, col+2] == self.board[row+3, col+3]:
                    return self.board[row,col]
        # Check diagonals (top-left to bottom-right)
        for row in range(3, self.row):
            for col in range(self.col - 3):
                if self.board[row, col] != 0 and self.board[row, col] == self.board[row-1, col+1] == self.board[row-2, col+2] == self.board[row-3, col+3]:
                    return self.board[row,col]

        return 0

    def LegalMoves(self):
        if self.IsTerminal(): return 0
        return np.sum(self.next < 6)
    

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
        reward = DefaultPolicy(leaf.state, turn)
        BackupNegamax(leaf, reward, turn)

    print(np.array([children.UCB(0, root.N) for children in root.children]))
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
    action = node.state.UntriedMove()
    #print(f'actio Expandn: {action}')
    new_state = copy.deepcopy(node.state)
    new_state.board[action[0]][action[1]] = turn
    new_state.next[action[1]] += 1
    node.AddNode(child_state=new_state)
    return node.children[-1]

def DefaultPolicy(state, turn) -> int:
    state = copy.deepcopy(state)
    iterations = 0
    while state.IsTerminal() is False and state.Winner() == 0:  
        action = state.UntriedMove()
        state.board[action[0]][action[1]] = turn
        state.next[action[1]] += 1
        turn = -turn
        iterations += 1
    return state.Winner()

def BackupNegamax(node, reward, turn):
    cnt = 0 
    while node is not None:
        cnt += 1
        node.N += 1
        node.Q -= turn*reward
        turn = -turn
        node = node.parent
    return 
    