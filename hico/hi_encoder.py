import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, kn_sz, sd, pd) -> None:
        super().__init__()
        self.dw_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=kn_sz, stride=sd, padding=pd, groups=in_ch)

    def forward(self, input_tensor):
        return self.dw_conv(input_tensor.transpose(2,1)).transpose(2,1)

class UDM(nn.Module):
    """Unified Donwsampling Module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, factor) -> None:
        super().__init__()
        self.conv = DepthWiseConv(in_channels,kernel_size,stride,padding)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(True)
        self.maxpooling = nn.MaxPool1d(factor)
        
    def forward(self, input_tensor):
        src = self.conv(input_tensor)
        src = self.norm(src)
        src = self.relu(src)
        src = self.maxpooling(src.transpose(2,1)).transpose(2,1)
        return src

class HiEncoder(nn.Module):
    """Two branch hierarchical encoder with multi-granularity"""

    def __init__(self, t_input_size, s_input_size, 
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 ) -> None:
        super().__init__()
        self.d_model  = hidden_size 
        self.granularity = granularity
        self.encoder = encoder
        
        # temproal and spatial branch embedding layers
        self.t_embedding = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        ) 
        self.s_embedding = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        )
        
        # downsampling modules
        self.t_downsample = UDM(hidden_size, hidden_size, kernel_size, stride, padding, factor)
        self.s_downsample = UDM(hidden_size, hidden_size, kernel_size, stride, padding, factor)

        # seq2seq encoders
        if encoder=="GRU":
            self.t_encoder = nn.GRU(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)
            self.s_encoder = nn.GRU(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)
        elif encoder=="LSTM":
            self.t_encoder = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)
            self.s_encoder = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)
        elif encoder=="Transformer":
            encoder_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first=True)
            self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
            self.s_encoder = TransformerEncoder(encoder_layer, num_layer)
        else:
            raise ValueError("Unknown encoder!")

    def forward(self, xc, xp):
        # Given the time-majored domain input sequence xc 
        # and the space-majored domain input sequence xp

        if self.encoder=="GRU" or self.encoder=="LSTM":
            self.t_encoder.flatten_parameters()
            self.s_encoder.flatten_parameters()
        

        # embedding
        xc = self.t_embedding(xc)
        xp = self.s_embedding(xp)


        #  two branch multi_granularity encoding
        if self.encoder=="GRU" or self.encoder=="LSTM":
            vc, _ = self.t_encoder(xc)
            vp, _ = self.s_encoder(xp)
        else:
            vc = self.t_encoder(xc)
            vp = self.s_encoder(xp)

        # implementation using amax for the TMP runs faster than using MaxPool1D
        # not support pytorch < 1.7.0
        vc = vc.amax(dim=1).unsqueeze(1)
        vp = vp.amax(dim=1).unsqueeze(1)
        
        
        for i in range(1, self.granularity):
            xc = self.t_downsample(xc)
            xp = self.s_downsample(xp)

            if self.encoder=="GRU" or self.encoder=="LSTM":
                vc_i, _ = self.t_encoder(xc)
                vp_i, _ = self.s_encoder(xp)
            else:
                vc_i = self.t_encoder(xc)
                vp_i = self.s_encoder(xp)

            vc_i = vc_i.amax(dim=1).unsqueeze(1)
            vp_i = vp_i.amax(dim=1).unsqueeze(1)
           
            vc = torch.cat([vc, vc_i], dim=1)
            vp = torch.cat([vp, vp_i], dim=1)
        
        return vc, vp


class PretrainingEncoder(nn.Module):
    """hierarchical encoder network + projectors"""

    def __init__(self, t_input_size, s_input_size, 
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer, 
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(PretrainingEncoder, self).__init__()
  
        self.d_model  = hidden_size 
        

        self.hi_encoder = HiEncoder(
            t_input_size, s_input_size,
            kernel_size, stride, padding, factor,
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # clip level feature projector
        self.clip_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # part level feature projector
        self.part_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # temporal domain level feature projector
        self.td_proj = nn.Sequential(
            nn.Linear(granularity*self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )
        
        # spatial domain level feature projector
        self.sd_proj = nn.Sequential(
            nn.Linear(granularity*self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )
        
        # instance level feature projector
        self.instance_proj = nn.Sequential(
            nn.Linear(2*granularity*self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, xc, xp):
        # we use concatenation as our feature fusion method

        # obtain clip and part level representations
        vc, vp = self.hi_encoder(xc, xp)

        # concatenate different granularity features as temproal and spatial domain representations
        vt = vc.reshape(vc.shape[0],-1)
        vs = vp.reshape(vp.shape[0],-1)

        # same for instance level representation
        vi = torch.cat([vt, vs], dim=1)

        # projection
        zc = self.clip_proj(vc)
        zp = self.part_proj(vp)

        zt = self.td_proj(vt)
        zs = self.sd_proj(vs)

        zi = self.instance_proj(vi)

        return zc, zp, zt, zs, zi
         
class DownstreamEncoder(nn.Module):
    """hierarchical encoder network + classifier"""

    def __init__(self, t_input_size, s_input_size, 
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer, 
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()
  
        self.d_model  = hidden_size 
        

        self.hi_encoder = HiEncoder(
            t_input_size, s_input_size,
            kernel_size, stride, padding, factor,
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # linear classifier
        self.fc = nn.Linear(2*granularity*self.d_model, num_class)

    def forward(self, xc, xp, knn_eval=False):
   
        vc, vp = self.hi_encoder(xc, xp)

        vt = vc.reshape(vc.shape[0],-1)
        vs = vp.reshape(vp.shape[0],-1)

        vi = torch.cat([vt, vs], dim=1)

        if knn_eval: # return last layer features during  KNN evaluation (action retrieval)
             return vi
        else:
             return self.fc(vi)