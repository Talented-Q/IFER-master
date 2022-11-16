import torch
import torch.nn as nn

mse = nn.MSELoss()

def D_loss(real_out, fake_out):
    loss_r = mse(real_out, torch.ones_like(real_out))
    loss_f = mse(fake_out, -torch.ones_like(fake_out))
    return loss_r + loss_f

def G_loss(fake_out):
    loss_f = mse(fake_out, torch.ones_like(fake_out))
    return loss_f