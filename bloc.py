import tkinter as tk
import numpy as np
from monte_carlo_connect4 import MCTS, Node, Board
import copy

class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.board = Board()
        self.turn = 1  # +1 for player, -1 for AI
        self.buttons = []
        self.canvas = None
        self.game_over = False
        
        self.create_widgets()
        self.display_board()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=700, height=600, bg='blue')
        self.canvas.pack()

        for col in range(7):
            button = tk.Button(self.root, text=f"Column {col}", command=lambda col=col: self.player_move(col))
            button.pack(side=tk.LEFT)
            self.buttons.append(button)

    def display_board(self):
        self.canvas.delete("all")  # Clear canvas

        display = {0: ".", 1: "X", -1: "O"}
        for row in range(6):
            for col in range(7):
                x1 = col * 100 + 10
                y1 = (5 - row) * 100 + 10
                x2 = col * 100 + 90
                y2 = (5 - row) * 100 + 90
                color = 'white' if self.board.board[row][col] == 0 else 'red' if self.board.board[row][col] == 1 else 'yellow'
                self.canvas.create_oval(x1, y1, x2, y2, fill=color)
        
    def player_move(self, col):
        if self.game_over or self.board.next[col] >= 6:
            return

        row = self.board.next[col]
        self.board.board[row][col] = self.turn
        self.board.next[col] += 1
        self.display_board()
        if self.check_winner():
            self.end_game("You win!")
        else:
            self.turn = -self.turn
            self.ai_move()

    def ai_move(self, budget=4000):
        if self.game_over:
            return

        root = Node(state=copy.deepcopy(self.board))
        best_node = MCTS(budget, root)
        action = np.argwhere(best_node.state.board != self.board.board)[0]
        col = action[1]
        print(f"AI places at column {col}.")
        
        row = self.board.next[col]
        self.board.board[row][col] = self.turn
        self.board.next[col] += 1
        self.display_board()
        if self.check_winner():
            self.end_game("AI wins!")
        else:
            self.turn = -self.turn

    def check_winner(self):
        winner = self.board.Winner()
        if winner != 0:
            return True
        return False

    def end_game(self, message):
        self.game_over = True
        self.canvas.create_text(350, 300, text=message, font=('Arial', 24), fill="white")
        for button in self.buttons:
            button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Connect 4")
    game = Connect4GUI(root)
    root.mainloop()
