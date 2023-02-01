import argparse
import time
import torch
import numpy as np
import os
import torch.optim as optim

# custom modules

# from loss import MonodepthLoss
# from utils import get_model, to_device, prepare_dataloader
import os
from PIL import Image
# plot params
import scipy.io
import torchvision.transforms as transforms

import tensorrt as trt

from transdssl.transdssl_encoder import TRANSDSSLEncoder
from transdssl.transdssl_decoder import TRANSDSSLDecoder
from transdssl.transdssl_multi import TransDSSL_multi
from transdssl.transdsslmodels_attn import TRANSDSSLDepthModel
model=TRANSDSSLDepthModel(backbone="S",infer=False)
from pytorch_quantization import nn as quant_nn
####################
enpre_model=torch.load("tmp/Kaist_sup_pwsa_v5/models/weights_30/encoder.pth")
enpre_keys=list(enpre_model.keys())

depre_model=torch.load("tmp/Kaist_sup_pwsa_v5/models/weights_30/depth.pth")
depre_keys=list(depre_model.keys())


encoder_st=model.state_dict()
encoder_st_key=list(encoder_st.keys())



for k in encoder_st_key:
    if k in enpre_keys:
        encoder_st[k]=enpre_model[k]
    if k in depre_keys:
        encoder_st[k]=depre_model[k]
model.load_state_dict(encoder_st)
# import pdb;pdb.set_trace()
#     encoder_st[k]=model[k]
# models["encoder"].load_state_dict(encoder_st)
####################

model=model.to("cuda")    
model.eval()
quant_nn.TensorQuantizer.use_fb_fake_quant = True
dummy_input=torch.randn((1,3,448,512)).cuda()
try:
    f="transdssl_v2.onnx"
    # torch.onnx.export(model,  
    #                   img,                
    #                   f,   
    #                   export_params=True, 
    #                   opset_version=13,          # 모델을 변환할 때 사용할 ONNX 버전
    #                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
    #                   input_names = ['image'],   # 모델의 입력값을 가리키는 이름
    #                   output_names = ['disp'], # 모델의 출력값을 가리키는 이름
    #                   dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
    #                                 'output' : {0 : 'batch_size'}})
    input_names = ["input_0"]
    output_names = ["output_0"]

    # Now with dynamic_axes, the output of TensorRT engine is wrong
    # So now we use fixed size
    dynamic_axes = {'input_0': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}}
    torch.onnx.export(model, dummy_input, f, verbose=False, opset_version=13,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      export_params=True,
                      do_constant_folding=True,
                      )
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)