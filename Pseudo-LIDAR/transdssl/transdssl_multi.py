import torch
import torch.nn as nn
import torch.nn.functional as F
from transdssl.transdssl_encoder import TRANSDSSLEncoder
from transdssl.transdssl_decoder import TRANSDSSLDecoder

class TransDSSL_multi(nn.Module):
    def __init__(self):
        super(TransDSSL_multi, self).__init__()
        self.models={}
        self.models["encoder"] =TRANSDSSLEncoder(backbone="S",infer=False)
        self.models["depth"] = TRANSDSSLDecoder(backbone="S",infer=False)
        self.models["color"] = TRANSDSSLDecoder(backbone="S",infer=False,color=True)
    def forward(self, x, epoch=0):
        # import pdb;pdb.set_trace()
        encoder_features=self.models["encoder"](x)
        disp=self.models["depth"](encoder_features)[('disp', 0)]
        color=self.models["color"](encoder_features)[('color_pred', 0)]
        return disp, color