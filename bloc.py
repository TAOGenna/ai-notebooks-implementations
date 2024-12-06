import torch
import torch.nn.functional as F

# Imagen de entrada: 1 canal, 3x3
input = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]]])

# Cuadrícula de muestreo
grid = torch.tensor([[[[-1.0, -1.0], [1.0, -1.0]],
                      [[-1.0,  1.0], [1.0,  1.0]]]])

# Muestreo
output = F.grid_sample(input, grid)
print(output)