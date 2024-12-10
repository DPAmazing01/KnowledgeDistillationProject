import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def summary(net):
    assert isinstance(net, nn.Module)
    print("Layer ID\tType\t\t# Of Parameters")
    layer_id = 0
    num_total_params = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            layer_id += 1
            print("%d\t\tLinear\t\t%d" %(layer_id, num_parameters))
            num_total_params += num_parameters
        elif isinstance(m, nn.Conv2d):
            weight = m.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            layer_id += 1
            print("%d\t\tConvolutional\t%d" % (layer_id, num_parameters))
            num_total_params += num_parameters
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            layer_id += 1
            print("%d\t\tBatchNorm\tN/A" % (layer_id))
        elif isinstance(m, nn.ReLU):
            layer_id += 1
            print("%d\t\tReLU\t\tN/A" % (layer_id))
    print("Total parameters: %d" %num_total_params)