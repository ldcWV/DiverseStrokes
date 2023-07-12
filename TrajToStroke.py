from torch import nn
from torch.nn import functional as F
import torch

class TrajToStroke(nn.Module):
    def __init__(self):
        super(TrajToStroke, self).__init__()
        
        self.conv_hidden_dims = [128, 64, 32, 16, 8]
        
        self.fc = nn.Sequential(
            nn.Linear(24, self.conv_hidden_dims[0]*2*2),
            nn.LeakyReLU()
        )
        
        decoder_layers = []
        for i in range(len(self.conv_hidden_dims) - 1):
            deconv_block = nn.Sequential(
                nn.ConvTranspose2d(self.conv_hidden_dims[i],
                                   self.conv_hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(self.conv_hidden_dims[i + 1]),
                nn.LeakyReLU()
            )
            decoder_layers.append(deconv_block)
        decoder_layers.append(nn.Sequential(
            nn.ConvTranspose2d(self.conv_hidden_dims[-1],
                               self.conv_hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.conv_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.conv_hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh()
        ))
        self.conv = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1, self.conv_hidden_dims[0], 2, 2)
        x = self.conv(x)
        return x
    
    def loss(self, stroke, stroke_hat):
        return F.mse_loss(stroke, stroke_hat)
    