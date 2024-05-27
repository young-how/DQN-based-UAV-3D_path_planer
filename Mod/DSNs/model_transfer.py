'''
Author: young how younghowkg@gmail.com
Date: 2023-10-10 15:26:33
LastEditors: younghow 1102708501@qq.com
LastEditTime: 2023-10-20 09:40:50
FilePath: \RLGF\Mod\DSNs\model_transfer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torch.distributions import Normal
from BaseCNN import *
if __name__=='__main__':
    root=os.path.dirname(os.path.abspath(__file__))   #当前目录名称
    input_dim=30
    hiden_dim=128
    out_dim=8
    param={'w':input_dim,'hiden_dim':hiden_dim,'output':out_dim}
    mod=PolicyNet_SAC(param)
    state_dict=torch.load(root+'\DSN_6_.pth')
    mod.load_state_dict(state_dict['model'])
    torch.save(mod,root+'\DSN_6.pth')
    #model_test=torch.load(root+'DSN_2.pth')
