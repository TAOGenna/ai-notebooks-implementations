import numpy as np

# action space of tictactoe is you have the option of putting a mark wherever there is a free cell 
# 
#
class State:
    """
    here is where things change depending on the problem 
    - define 
    """
    def __init__(self):
        self.arr = np.zeros((3,3))
    def terminal(self):
        cnt = 0
        for i in range(3):
            for j in range(3):
                if self.arr[i][j] != 0:
                    cnt +=1 
        return cnt == 0 

class Node:
    def __init__(self, state = State(), action = None, parent = None):
        # node as a state holder properties 
        self.state    = state
        self.action   = action
        self.Q = 0 # Q(v): total reward of all playouts that passed through this state
        self.N = 0 # N(v): number of times it has been visited
        # graph stuff
        self.childs = [] # array of nodes
        self.parent = parent
    
    def AddNode(self, state: State, action, parent):
        self.childs.append(Node(state=state, action=action, parent=parent))
        return self.childs[-1]
    
    
# TODO: in what moment are we finishing the game internally to see possible results? 
# TODO: delta(v,p): denotes the component of the reward vector ∆ associated with the current
# player p at node v.

# implementation of the UCT algorithm
# budget is number of iterations I guess
BUDGET = 10
def UCTSearch(s0: State):
    root = Node(state=s0)
    
    while BUDGET:
        # expansion fase 
        vl = TreePolicy(root)
        # simulation fase
        delta = DefaultPolicy(vl.state)
        # backprop fase
        backup(vl,delta)
        BUDGET -= 1 
    # The return value of the overall search in this case is
    # a(BESTCHILD(v0, 0)) which will give the action `a` that leads to the child with the highest reward
    return BestChild(root,0)

def TreePolicy(v: Node):
    Cp = 1.0/np.sqrt(2)
    while v.state.terminal() is False :
        if v.state.terminal() is False : 
            return Expand(v)
        else:
            v = BestChild(v,Cp)
    return v

def SelectUntriedAction(state: State):
    actions = [] 
    for i in range(3):
        for j in range(3):
            if state.arr[i][j] == 0:
                actions.append((i,j))
    return np.random.choice(actions)

def PerformAction(state, action):
    foo = state # REVISIT
    foo.arr[action[0]][action[1]] = -1
    return foo

def Expand(v: Node):
    action = SelectUntriedAction(v.state)
    state = PerformAction(state, action)
    return v.AddNode(state=state, action=action, parent=v)
     
def BestChild(v: Node, c):
    arr = np.array([])
    for node in v.childs:
        uct = node.Q/node.N + c*np.sqrt(2*np.log(node.N)/node.N)
        arr = np.append(arr,uct)
    return v.childs[np.argmax(arr)]

def DefaultPolicy(s: State):
    state = s
    while state.terminal() is False:
        # action selected at random
        action = SelectUntriedAction(state) 
        state = PerformAction(state,action)
    return # TODO: REWARD 

def backup(v: Node, delta):
    node = v 
    while v is not None:
        v.B += 1
        v.Q += delta(node,p)
        node = node.parent

#--------------------------------------------------------------------------------------------------

# DEFINE GAME 

# always the first turn is mine
game = State()
# mark 1 is mine, and -1 for the computer
game_finished = False
while game_finished is not True:
    # print current state of the game 
    print('Current state of the game:')
    print(game.arr)

    # user enters a response
    cellx, celly = map(int, input("Enter two numbers separated by space: ").split())
    game.arr[cellx][celly] = +1
    
    # compute MCTS answers
    answer = UCTSearch(game)
    print('mcts response is ( {answer.x} , {answer.y} ) ')
    game.arr[answer.x][answer.y] = -1
    if game.terminal():
        print('GAME FINISHED')
        break
