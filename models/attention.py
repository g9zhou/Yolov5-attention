import torch
from torch import nn
from models.common import *

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, act):
        super(MultiHeadSelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.act = act
        
        self.q = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.k = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.v = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        b, C, w, h = x.size()
        proj_q  = self.q(x).view(b,-1,w*h).permute(0,2,1) # B X CX(N)
        proj_k =  self.k(x).view(b,-1,w*h) # B X C x (*W*H)
        energy =  torch.bmm(proj_q,proj_k) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_v = self.v(x).view(b,-1,w*h) # B X C X N

        out = torch.bmm(proj_v,attention.permute(0,2,1) )
        out = out.view(b,C,w,h)
        
        out = self.gamma*out + x
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer):
        super().__init()
        self.conv = None
        if (in_dim != out_dim):
            self.conv = Conv(in_dim, out_dim)
        self.linear = nn.Linear(in_dim,out_dim)
        self.block = nn.Sequential(*(MultiHeadSelfAttention(in_dim) for _ in range(num_layer)))
        self.out = out_dim

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2,0,1)
        return self.block(p + self.linear(p)).permute(1,2,0).reshape(b, self.out, w, h)


# class BottleneckAttention(nn.Module):
#     def __init__(self, in_dim, out_dim, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         hidden = int(out_dim*e)
#         self.cv1 = Conv(in_dim, hidden, 1, 1)
#         self.cv2 = Conv(hidden, out_dim, 3, 1, g=g)
#         self.add = shortcut and in_dim==out_dim

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Attention(nn.Module):
    def __init__(self, in_dim, out_dim, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = int(out_dim*e)
        self.cv1 = Conv(in_dim, hidden, 1, 1)
        self.cv2 = Conv(in_dim, hidden, 1, 1)
        self.cv3 = Conv(2*hidden, out_dim, 1)
        self.m = AttentionBlock(hidden, hidden, shortcut, g, e=1.0)

    def forward(self,x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

