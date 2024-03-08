import torch
import torch.nn as nn

class TSMixer(nn.Module):
 
    def __init__(self, enc_in, num_blocks,
                 hidden_size, seq_len, pred_len, 
                 individual, dropout, activation, 
                 single_layer_mixer):
        
        super(TSMixer, self).__init__()
        self.num_blocks = num_blocks
    
        self.mixer_block = MixerBlock(enc_in, hidden_size,
                                      seq_len, dropout,
                                        activation, single_layer_mixer)
        self.channels = enc_in
        self.pred_len = pred_len
        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(self.channels, pred_len))
        else :
            self.output_linear_layers = nn.Linear(self.channels, pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        #Final linear layer applied on the transpoed mixers' output

        # x = torch.swapaxes(x, 1, 2)
        # print(f'mixer input swaped shape: {x.shape}')
        # #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        if self.individual_linear_layers :
            for c in range(self.channels): 
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else :
            y = self.output_linear_layers(x.clone())
        

        y = torch.swapaxes(y, 1, 2)
       
        return y

    
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer :
            self.linear_layer1 = nn.Linear(channels, channels)
        else :
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        if not(self.single_layer_mixer) :
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y
        

    
class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y
    
    
class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)
    
    def forward(self, x) :
        y = self.timesteps_mixer(x)   
        y = self.channels_mixer(y)
        return y