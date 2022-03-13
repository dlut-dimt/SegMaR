import mobula
import torch
import torch.nn.functional as F
from attention_sampler import mobula_hack_for_pytorch

#from .attention_sampler import AttSamplerGrid
# Hack MobulaOP for the compatible functions
torch._mobula_hack = mobula_hack_for_pytorch
mobula.op.load('attention_sampler')

class AttSampler(torch.nn.Module):
    def __init__(self, scale=1.0, dense=4, iters=1):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):
        grid = mobula.op.AttSamplerGrid(data.detach(),
                                        attx.detach(),
                                        atty.detach())
        grid = torch.stack(grid, dim=3)
        #print('data_data',data.shape)
        #print('grid_grid',grid.shape)
        #print('grid',grid)
        return F.grid_sample(data, grid), grid
