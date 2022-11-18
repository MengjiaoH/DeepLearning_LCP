import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    
    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Encoder(nn.Module):
    def __init__(self, activation=None):
        super(Encoder, self).__init__()
        w0 = 50
        
        self.mean_layer = nn.Sequential(
            Siren(4, 32, w0 = w0, c = 6, is_first = True, use_bias = True, activation = activation),
            Siren(32, 64, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(64, 128, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(128, 256, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(256, 512, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(512, 1024, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(1024, 1024, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
         )
        self.iso_layer = nn.Sequential(
            Siren(1, 32, w0 = w0, c = 6, is_first = True, use_bias = True, activation=activation),
            Siren(32, 64, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(64, 128, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            # Siren(128, 256, w0 = 1., c = 6, is_first = False, use_bias = True),
            # Siren(256, 512, w0 = 1., c = 6, is_first = False, use_bias = True),
            # Siren(512, 1024, w0 = 1., c = 6, is_first = False, use_bias = True),
            # Siren(1024, 1024, w0 = 1., c = 6, is_first = False, use_bias = True),
         )

        self.cov_layer = nn.Sequential(
            Siren(10, 32, w0 = w0, c = 6, is_first = True, use_bias = True, activation=activation),
            Siren(32, 64, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(64, 128, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(128, 256, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(256, 512, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(512, 1024, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(1024, 1024, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
         ) 
        
        self.decode = nn.Sequential(
            Siren(2048+128, 2048, w0 = w0, c = 6, is_first = True, use_bias = True, activation=activation),
            Siren(2048, 1024, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(1024, 512, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(512, 256, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(256, 128, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(128, 64, w0 = 1., c = 6, is_first = False, use_bias = True, activation=activation),
            Siren(64, 1, w0 = 1., c = 6, is_first = False, use_bias = True, activation=nn.Sigmoid()),
         )     
    def forward(self, m, v, iso):
        mean_latent = self.mean_layer(m)
        cov_latent = self.cov_layer(v)
        iso_latent = self.iso_layer(iso)
        x = torch.cat((mean_latent, cov_latent, iso_latent), axis=1)
        x = self.decode(x)
        return x

if __name__ == "__main__":
    model = Encoder()
    p = torch.rand(2, 4)
    t = torch.rand(2, 4)
    x = model(p, t)
    print("x.shape", x.shape)
