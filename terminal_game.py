import numpy as np
from monte_carlo_connect4 import MCTS, Node, Board
import copy

class Connect4Terminal:
    def __init__(self):
        self.board = Board()
        self.turn = 1  # +1 for player, -1 for AI

    def display_board(self):
        display = {0: ".", 1: "X", -1: "O"}
        print("\n  0 1 2 3 4 5 6")
        for row in self.board.board[::-1]:  # Print from top to bottom
            print(" ", " ".join(display[cell] for cell in row))
        print()

    def player_move(self):
        while True:
            try:
                col = int(input("Enter the column (0-6): "))
                if col < 0 or col >= 7 or self.board.next[col] >= 6:
                    raise ValueError("Invalid move. Try again.")
                break
            except ValueError as e:
                print(e)

        row = self.board.next[col]
        self.board.board[row][col] = self.turn
        self.board.next[col] += 1

    def ai_move(self, budget=4000):
        root = Node(state=copy.deepcopy(self.board))
        best_node = MCTS(budget, root)
        action = np.argwhere(best_node.state.board != self.board.board)[0]
        print(f"AI places at column {action[1]}.")
        self.board = copy.deepcopy(best_node.state)

    def play(self):
        print("Welcome to Connect 4! You are 'X', and the AI is 'O'.")
        self.display_board()

        while not self.board.IsTerminal():
            if self.turn == 1:
                self.player_move()
            else:
                self.ai_move()

            self.display_board()
            winner = self.board.Winner()
            if winner != 0:
                print("You win!" if winner == 1 else "AI wins!")
                return
            self.turn = -self.turn

        print("It's a draw!")

if __name__ == "__main__":
    game = Connect4Terminal()
    game.play()
