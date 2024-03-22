import torch
import torch.nn as nn

import layer_confs as lcfg

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(DownsampleBlock, self).__init__()
        # Design - Maxpooling -> Conv2d & ReLU -> Conv2d & ReLU
        
        # All the MaxPooling and Upsample2D are size (2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=lcfg.POOLING_KERNEL_SIZE)
        
        # Assuming that we follow the convention of increasing the number of channels
        # in the first conv layer of the block as Unet does.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size = lcfg.CONV_KERNEL_SIZE, 
            stride = lcfg.CONV_STRIDE,
            padding = 'same',
            )
        self.conv1_relu = nn.ReLU()
        
        if use_batchnorm:
            self.conv1_bn = nn.BatchNorm2d(out_channels)
            
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size = lcfg.CONV_KERNEL_SIZE, 
            stride = lcfg.CONV_STRIDE,
            padding = 'same',
            )
        
        self.conv2_relu = nn.ReLU()
        if use_batchnorm:
            self.conv2_bn = nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        if hasattr(self, 'conv1_bn'):
            x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv2_relu(x)
        if hasattr(self, 'conv2_bn'):
            x = self.conv2_bn(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = False) -> None:
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels//2,
            kernel_size = lcfg.UPSAMPLE_KERNEL_SIZE,
            stride = lcfg.UPSAMPLE_KERNEL_SIZE
            )
        self.conv1 = nn.Conv2d(
            in_channels + in_channels//2, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
    def forward(self, x, skipped_match):
        x = self.upsample(x)
        x = torch.cat([x, skipped_match], dim=1)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        return x
    
if __name__  == "__main__":
    import torch
    
    DUMP_ONNX = False
    
    print("=================================================")

    # Basic connectivity sanity check
    # Test DownsampleBlock
    downsample_block = DownsampleBlock(4, 4)
    
    ds1_input = torch.randn(1, 4, 256, 384)
    ds1_output = downsample_block(ds1_input)
    print("DownsampleBlock input shape : ", ds1_input.shape)
    print("DownsampleBlock output shape : ", ds1_output.shape)
    if DUMP_ONNX:
        torch.onnx.export(downsample_block, ds1_input, "downsample_block.onnx")
        
    print("=================================================")
        
    # Test UpsampleBlock
    upsample_block = UpsampleBlock(4, 4)
    us1_output = upsample_block(ds1_output, ds1_input)
    print("UpsampleBlock input shapes : ", ds1_output.shape, ds1_input.shape)
    print("UpsampleBlock output shape : ", us1_output.shape)
    if DUMP_ONNX:
        torch.onnx.export(upsample_block, (ds1_output, ds1_input), "upsample_block.onnx")
        
    print("=================================================")
        