import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class linearModel(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, enc_in, individual, seq_len, pred_len):
        super(linearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,i,:] = self.Linear[i](x[:,i,:].clone())
            # x = output
        else:
            # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            x = self.Linear(x.clone())
        x = torch.swapaxes(x, 1, 2) 
        return x # [Batch, Output length, Channel]
    
    
        #     y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        # if self.individual_linear_layers :
        #     for c in range(self.channels): 
        #         y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        # else :
        #     y = self.output_linear_layers(x.clone())
        

        # y = torch.swapaxes(y, 1, 2)
       
        # return y