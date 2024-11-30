import tkinter as tk
import numpy as np
from monte_carlo_connect4 import MCTS, Node, Board
import copy

ROWS, COLS = 6, 7
CIRCLE_RADIUS = 40
PADDING = 10
GRID_COLOR = "#b3b3b3"
CIRCLE_COLOR = "#ffffff"

class Connect4Interface:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=COLS * (CIRCLE_RADIUS * 2 + PADDING),
                                height=ROWS * (CIRCLE_RADIUS * 2 + PADDING),
                                bg=GRID_COLOR)
        self.canvas.pack()
        self.state = Board()  # Use state as a class attribute
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.player_move)

    def draw_grid(self):
        for row in range(ROWS):
            for col in range(COLS):
                x0 = col * (CIRCLE_RADIUS * 2 + PADDING) + PADDING
                y0 = row * (CIRCLE_RADIUS * 2 + PADDING) + PADDING
                x1 = x0 + CIRCLE_RADIUS * 2
                y1 = y0 + CIRCLE_RADIUS * 2
                self.canvas.create_oval(x0, y0, x1, y1, fill=CIRCLE_COLOR, outline="")

    def player_move(self, event):
        col = event.x // (CIRCLE_RADIUS * 2 + PADDING)
        if col < 0 or col >= COLS: return
        row = self.find_empty_row(col)
        if row is not None:
            self.place_token(row, col, 1)
            if self.state.IsTerminal():
                self.show_winner(self.state.Winner())
            else:
                self.ai_move()

    def find_empty_row(self, col):
        for row in range(ROWS - 1, -1, -1):
            if self.state.board[row, col] == 0:
                return row
        return None

    def place_token(self, row, col, player):
        self.state.board[row, col] = player
        color = "#ff0000" if player == 1 else "#0000ff"
        x0 = col * (CIRCLE_RADIUS * 2 + PADDING) + PADDING
        y0 = row * (CIRCLE_RADIUS * 2 + PADDING) + PADDING
        x1 = x0 + CIRCLE_RADIUS * 2
        y1 = y0 + CIRCLE_RADIUS * 2
        self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="")

    def ai_move(self):
        budget = 500
        best_node = MCTS(budget, Node(state=copy.deepcopy(self.state)))

        move = None
        for col in range(COLS):
            if best_node.state.next[col] != self.state.next[col]:
                move = col
                break

        if move is not None:
            row = self.find_empty_row(move)  # Use find_empty_row
            self.place_token(row, move, -1)
            self.state = best_node.state  # Update the state
            if self.state.IsTerminal():
                self.show_winner(self.state.Winner())

    def show_winner(self, winner):
        if winner == 1:
            message = "Player wins!"
        elif winner == -1:
            message = "AI wins!"
        else:
            message = "It's a draw!"
        self.canvas.create_text(self.canvas.winfo_width() // 2, 
                                self.canvas.winfo_height() // 2, 
                                text=message, font=("Arial", 24), fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Connect 4")
    app = Connect4Interface(root)
    root.mainloop()
