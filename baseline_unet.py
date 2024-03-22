import torch
import torch.nn as nn

import layer_confs as lcfg

from blocks import DownsampleBlock, UpsampleBlock, DoubleUpsampleBlock, DoubleDownSampleBlock

class UnetBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=lcfg.CONV_KERNEL_SIZE, stride=lcfg.CONV_STRIDE, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=lcfg.CONV_KERNEL_SIZE, stride=lcfg.CONV_STRIDE, padding='same'),
            nn.ReLU()
            )
        
        self.dsb1 = DownsampleBlock(4, 4)
        self.ddsb1 = DoubleDownSampleBlock(4, 8)
        self.ddsb2 = DoubleDownSampleBlock(8, 16)
        
        self.dupsb1 = DoubleUpsampleBlock(16, 8)
        self.dupsb2 = DoubleUpsampleBlock(8, 4)
        self.upsb3 = UpsampleBlock(4, 4)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=lcfg.CONV_KERNEL_SIZE, stride=lcfg.CONV_STRIDE, padding='same'),
            nn.ReLU(),
            )
        
    def forward(self, x):
        print("UNET : input shape : ", x.shape)
        x = self.input_conv(x)
        skip1 = x
        print("UNET : input_conv shape : ", x.shape)
        
        l1 = self.dsb1(x)
        print("UNET : dsb1 shape : ", l1.shape)
        
        x, skip2, skip3 = self.ddsb1(l1)
        print("UNET : ddsb1 shape : ", x.shape, skip2.shape, skip3.shape)
        
        x, skip4, skip5 = self.ddsb2(x)
        print("UNET : ddsb2 shape : ", x.shape, skip4.shape, skip5.shape)
        
        x = self.dupsb1(x, skip5, skip4)
        print("UNET : dupsb1 shape : ", x.shape)
        
        x = self.dupsb2(x, skip3, skip2)
        print("UNET : dupsb2 shape : ", x.shape)
        
        x = self.upsb3(x, skip1)
        print("UNET : upsb3 shape : ", x.shape)
        
        x = self.output_conv(x)
        print("UNET : output_conv shape : ", x.shape)
        
        return x
    
if __name__ == "__main__":
    model = UnetBaseline()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Output shape : ", y.shape)
    
    assert y.shape == x.shape
    
    DUMP_ONNX = True
    
    if DUMP_ONNX:
        torch.onnx.export(model, x, "baseline_unet.onnx", verbose=True)
        print("ONNX model dumped.")