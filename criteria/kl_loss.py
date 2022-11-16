import torch
import torch.nn as nn

kl_div = nn.KLDivLoss()
sfm = nn.Softmax()

def KL_loss(t_style,s_style,t_constant,s_constant):
    t_style = sfm(t_style)
    s_style = sfm(s_style)
    t_constant = sfm(t_constant)
    s_constant = sfm(s_constant)
    style_kl = kl_div(torch.log(t_style), s_style)
    constant_kl = kl_div(torch.log(t_constant), s_constant)
    klloss = style_kl + constant_kl
    return klloss

# style1 = torch.rand(size=(2,18,512))
# style2 = torch.rand(size=(2,18,512))
# constant1 = torch.rand(size=(2,512,4,4))
# constant2 = torch.rand(size=(2,512,4,4))
# loss = KL_loss(style1,style2,constant1,constant2)
# print(loss)