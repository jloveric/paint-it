import pytest
from dc_pbr.skip import skip
import torch

def test_skip():
        
    input_depth = 3
    net = skip(input_depth, 9,
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[128] * 5,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
    
    sample = torch.rand(1,3,1024,1024).type(torch.cuda.FloatTensor)

    res = net(sample)
    assert res.shape  == torch.Size([1,9,1024,1024])
    print('res', res.shape)