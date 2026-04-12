import arcade
import random
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ConvolutionNeuralNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import torch

CNN = ConvolutionNeuralNetwork()
CNN.load_state_dict(torch.load("models/cnn/CNN.pth", map_location=device))
CNN.to(device)
CNN.eval()

WIDTH = 280
HEIGHT = 280
ROWS = 28
COLS = 28
SQUARE_SIZE = WIDTH // COLS

WHITE = arcade.color.WHITE



class Zahlenerkennen(arcade.Window):

    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Schach KI")
        self.black_square = None

        self.grid = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]
        
        

    def on_draw(self):
        arcade.start_render()

        for row in range(ROWS):
            for col in range(COLS):

                color = (self.grid[row][col],self.grid[row][col],self.grid[row][col])
                
                if self.grid[row][col] > 255.0:
                    self.grid[row][col] = 255.0

                left = col * SQUARE_SIZE
                bottom = HEIGHT - (row + 1) * SQUARE_SIZE

                arcade.draw_lrtb_rectangle_filled(
                    left,
                    left + SQUARE_SIZE,
                    bottom + SQUARE_SIZE,
                    bottom,
                    
                    color
                )

        

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        
        
        if buttons & arcade.MOUSE_BUTTON_LEFT:
            

            col = int(x // SQUARE_SIZE)
            row = ROWS - 1 - int(y // SQUARE_SIZE)

            if 0 <= row < ROWS and 0 <= col < COLS:
                self.grid[row][col] += random.randint(80, 120)
                if row != 0:
                    self.grid[row-1][col] += random.randint(20, 50)
                if row != ROWS-1:
                    self.grid[row+1][col] += random.randint(20, 50)
                if col != 0:
                    self.grid[row][col-1] += random.randint(20, 50)
                if col != COLS-1:
                    self.grid[row][col+1] += random.randint(20, 50)
               
     
       
        
    

    def ReLU(self, x):
        return max(0, x)


    def softmax(self, z_out):
        max_z = max(z_out)
        exp_values = [math.exp(z - max_z) for z in z_out]
        sum_exp = sum(exp_values)
        return [v/sum_exp for v in exp_values] 

            
            
    def pred(self):
        grid = np.array(self.grid, dtype=np.float32) / 255.0
        tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = CNN(tensor)
            pred = torch.argmax(output, dim=1).item()

        print("Prediction:", pred)

    def reset(self):
        self.grid = [[0.0 for _ in range(ROWS)] for _ in range(COLS)]

        
    
    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.pred()
        elif key == arcade.key.ENTER:
            self.reset()

    
        




if __name__ == "__main__":
    game = Zahlenerkennen()
    arcade.run()
    

