import torch
import torch.nn as nn
import warnings

from TSConv import TSConv

import layer_confs as lcfg
from blocks import DownsampleBlock, UpsampleBlock, DoubleUpsampleBlock, DoubleDownSampleBlock, validate_conv_type

class Unet(nn.Module):
    
    convOptions = {
        "Conv2d": nn.Conv2d,
        "vanilla": nn.Conv2d, # Alias for Conv2d
        "baseline": nn.Conv2d, # Alias for Conv2d
        "TSConv": TSConv,
        "tsconv": TSConv
    }
    
    def __init__(self, conv : nn.Module = nn.Conv2d, use_batchnorm = False) -> None:
        super().__init__()
        
        validate_conv_type(conv)
        
        self.input_conv = nn.Sequential(
            conv(1, 4, kernel_size=lcfg.CONV_KERNEL_SIZE, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4) if use_batchnorm else nn.Sequential(),
            conv(4, 4, kernel_size=lcfg.CONV_KERNEL_SIZE, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4) if use_batchnorm else nn.Sequential()
            )
        
        self.dsb1 : DownsampleBlock = DownsampleBlock(4, 4, conv = conv, use_batchnorm=use_batchnorm)
        self.ddsb1 : DoubleDownSampleBlock = DoubleDownSampleBlock(4, 8, conv = conv, use_batchnorm=use_batchnorm)
        self.ddsb2 : DoubleDownSampleBlock = DoubleDownSampleBlock(8, 16, conv = conv, use_batchnorm=use_batchnorm)
        
        self.dupsb1 : DoubleUpsampleBlock = DoubleUpsampleBlock(16, 8, conv = conv, use_batchnorm=use_batchnorm)
        self.dupsb2 : DoubleUpsampleBlock = DoubleUpsampleBlock(8, 4, conv = conv, use_batchnorm=use_batchnorm)
        self.upsb3 : UpsampleBlock = UpsampleBlock(4, 4, conv = conv, use_batchnorm=use_batchnorm)
        
        self.output_conv = nn.Sequential(
            conv(4, 1, kernel_size=lcfg.CONV_KERNEL_SIZE, padding='same'),
            nn.ReLU(),
            )
        
    @torch.jit.ignore
    def check_tensor_dims(self, x):
        if isinstance(x, torch.Tensor):
            if x.shape[2] < 2 or x.shape[3] < 2:
                warnings.warn("Tensor dimensions are too small. Check architecture.")
        elif isinstance(x, list) or isinstance(x, tuple):
            for t in x:
                if t.shape[2] < 2 or t.shape[3] < 2:
                    warnings.warn("Tensor dimensions are too small. Check architecture.")
        else:
            pass
        
    def forward(self, x):
        x = self.input_conv(x)
        skip1 = x
        
        l1 = self.dsb1(x)
        
        x, skip2, skip3 = self.ddsb1(l1)
        self.check_tensor_dims((x, skip2, skip3))
        
        x, skip4, skip5 = self.ddsb2(x)
        self.check_tensor_dims((x, skip4, skip5))
        
        x = self.dupsb1(x, skip5, skip4)
        
        x = self.dupsb2(x, skip3, skip2)
        
        x = self.upsb3(x, skip1)
        
        x = self.output_conv(x)
        
        return x
    
if __name__ == "__main__": # pragma: no cover
    # CONV_MODE = nn.Conv2d
    CONV_MODE = TSConv
    
    BATCHNORM = False
    
    model = Unet(
        conv=CONV_MODE,
        use_batchnorm=BATCHNORM
        )
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Output shape : ", y.shape)
    
    assert y.shape == x.shape
    
    DUMP_ONNX = True
    
    if DUMP_ONNX:
        
        if CONV_MODE is TSConv:
            raise ValueError("Cannot export complete model to ONNX with TSConv")
        
        torch.onnx.export(model, x, "baseline_unet.onnx", verbose=True)
        print("ONNX model dumped.")