'''
Author: young how younghowkg@gmail.com
Date: 2023-10-10 16:01:13
LastEditors: young how younghowkg@gmail.com
LastEditTime: 2023-10-10 16:01:57
FilePath: \RLGF\Mod\DSNs\load_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torch.distributions import Normal
root=os.path.dirname(os.path.abspath(__file__))   #当前目录名称
mod=torch.load(root+'\DSN_0.pth')