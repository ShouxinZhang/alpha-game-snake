import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class SnakeNet(nn.Module):
    def __init__(self, grid_width, grid_height, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Input: 4 channels
        # 0: Body
        # 1: Head
        # 2: Food
        # 3: Hunger (normalized)
        self.conv_input = nn.Sequential(
            nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * grid_width * grid_height, 4) # 4 actions: Up, Down, Left, Right
            # Softmax will be applied in loss function (CrossEntropy)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * grid_width * grid_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
