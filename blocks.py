import torch
import torch.nn as nn

from TSConv import TSConv
import layer_confs as lcfg

def validate_conv_type(conv):
    if (not isinstance(conv, (type))) or (type(conv) in [nn.Conv2d, TSConv]):
        raise TypeError("Provided with a non-class type. Acceptable : nn.Conv2d and TSConv. Given : " + str(conv))

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv : nn.Module = nn.Conv2d, use_batchnorm=False):
        """Downsample block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            conv (Union[nn.Conv2D, TSConv]): Type of convolution to use
            use_batchnorm (bool, optional): Toggle batchnorm. Defaults to False.
        """
        super(DownsampleBlock, self).__init__()
        # Design - Maxpooling -> Conv2d & ReLU -> Conv2d & ReLU
        
        validate_conv_type(conv)
        
        # All the MaxPooling and Upsample2D are size (2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=lcfg.POOLING_KERNEL_SIZE)
        
        # Assuming that we follow the convention of increasing the number of channels
        # in the first conv layer of the block as Unet does.
        self.conv1 = conv(
            in_channels, out_channels, 
            kernel_size = lcfg.CONV_KERNEL_SIZE, 
            stride = lcfg.CONV_STRIDE,
            padding = 'same',
            )
        self.conv1_relu = nn.ReLU()
        
        if use_batchnorm:
            self.conv1_bn = nn.BatchNorm2d(out_channels)
            
        self.conv2 = conv(
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

class DoubleDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv : nn.Module = nn.Conv2d, use_batchnorm = False):
        """Double downsample block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            conv (nn.Module): Convolution type
            use_batchnorm (bool, optional): Toggle batchnorm. Defaults to False.
        """
        super().__init__()
        
        validate_conv_type(conv)
        
        self.maxpool = nn.MaxPool2d(kernel_size=lcfg.POOLING_KERNEL_SIZE)
        self.l1_conv1 = conv(
            in_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.l1_conv2 = conv(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
        self.l2_conv1 = conv(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.l2_conv2 = conv(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.bn4 = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        skip1 = x
        x = self.maxpool(x)
        x = self.l1_conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.l1_conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        skip2 = x
        x = self.maxpool(x)
        x = self.l2_conv1(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.l2_conv2(x)
        if self.use_batchnorm:
            x = self.bn4(x)
        return x, skip1, skip2

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv : nn.Module = nn.Conv2d, use_batchnorm = False) -> None:
        """Upsample block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            use_batchnorm (bool, optional): Toggle batchnorm. Defaults to False.
        """
        super().__init__()
        
        validate_conv_type(conv)
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels//2,
            kernel_size = lcfg.UPSAMPLE_KERNEL_SIZE,
            stride = lcfg.UPSAMPLE_KERNEL_SIZE
            )
        self.conv1 = conv(
            in_channels + in_channels//2, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.conv2 = conv(
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
    
class DoubleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv : nn.Module = nn.Conv2d, use_batchnorm = False) -> None:
        """Double upsample block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            use_batchnorm (bool, optional): Toggle batchnorm. Defaults to False.
        """
        super().__init__()
        
        validate_conv_type(conv)
        
        self.upsample1 = nn.ConvTranspose2d(
            in_channels, in_channels//2,
            kernel_size = lcfg.UPSAMPLE_KERNEL_SIZE,
            stride = lcfg.UPSAMPLE_KERNEL_SIZE
            )
        self.l1_conv1 = conv(
            in_channels + in_channels//2, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.l1_conv2 = conv(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
        self.upsample2 = nn.ConvTranspose2d(
            out_channels, out_channels//2,
            kernel_size = lcfg.UPSAMPLE_KERNEL_SIZE,
            stride = lcfg.UPSAMPLE_KERNEL_SIZE
            )
            
        self.l2_conv1 = conv(
            out_channels + out_channels//2, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        self.l2_conv2 = conv(
            out_channels, out_channels,
            kernel_size = lcfg.CONV_KERNEL_SIZE,
            stride = lcfg.CONV_STRIDE,
            padding = 'same'
            )
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.bn4 = nn.BatchNorm2d(out_channels)
            
    def forward(self, x, skipped_match_1, skipped_match_2):
        x = self.upsample1(x)
        print("DOUBLE UPSAMPLE BLOCK : upsampled shape : ", x.shape)
        print("DOUBLE UPSAMPLE BLOCK : skipped_match_1 shape : ", skipped_match_1.shape)
        print("DOUBLE UPSAMPLE BLOCK : skipped_match_2 shape : ", skipped_match_2.shape)
        x = torch.cat([x, skipped_match_1], dim=1)
        x = self.l1_conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.l1_conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
            
        x = self.upsample2(x)
        x = torch.cat([x, skipped_match_2], dim=1)
        x = self.l2_conv1(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.l2_conv2(x)
        if self.use_batchnorm:
            x = self.bn4(x)
        return x
 
if __name__  == "__main__":
    import torch
    
    DUMP_ONNX = False
    # CONV_MODE = nn.Conv2d
    CONV_MODE = TSConv
    
    print("=================================================")

    # Basic connectivity sanity check
    # Test DownsampleBlock
    downsample_block = DownsampleBlock(4, 4, conv=CONV_MODE)
    
    ds1_input = torch.randn(1, 4, 256, 384)
    ds1_output = downsample_block(ds1_input)
    print("DownsampleBlock input shape : ", ds1_input.shape)
    print("DownsampleBlock output shape : ", ds1_output.shape)
    if DUMP_ONNX:
        torch.onnx.export(downsample_block, ds1_input, "downsample_block.onnx")
        
    print("=================================================")
    
    # Test DoubleDownSampleBlock
    double_downsample_block = DoubleDownSampleBlock(4, 8, conv=CONV_MODE)
    dds1_output, dds1_skip1, dds1_skip2 = double_downsample_block(ds1_output)
    print("DoubleDownSampleBlock input shape : ", ds1_output.shape)
    print("DoubleDownSampleBlock output shape : ", dds1_output.shape)
    print("DoubleDownSampleBlock skip1 shape : ", dds1_skip1.shape)
    print("DoubleDownSampleBlock skip2 shape : ", dds1_skip2.shape)
    if DUMP_ONNX:
        torch.onnx.export(double_downsample_block, ds1_output, "double_downsample_block.onnx")
        
    print("=================================================")
    
    # Test DoubleUpsampleBlock
    double_upsample_block = DoubleUpsampleBlock(8, 4, conv=CONV_MODE)
    dus1_output = double_upsample_block(dds1_output, dds1_skip2, dds1_skip1)
    print("DoubleUpsampleBlock input shapes : ", dds1_output.shape, dds1_skip1.shape, dds1_skip2.shape)
    print("DoubleUpsampleBlock output shape : ", dus1_output.shape)
    if DUMP_ONNX:
        torch.onnx.export(double_upsample_block, (dds1_output, dds1_skip2, dds1_skip1), "double_upsample_block.onnx")
        
    print("=================================================")
        
    # Test UpsampleBlock
    upsample_block = UpsampleBlock(4, 4, conv = CONV_MODE)
    us1_output = upsample_block(ds1_output, ds1_input)
    print("UpsampleBlock input shapes : ", ds1_output.shape, ds1_input.shape)
    print("UpsampleBlock output shape : ", us1_output.shape)
    if DUMP_ONNX:
        torch.onnx.export(upsample_block, (ds1_output, ds1_input), "upsample_block.onnx")
        
    print("=================================================")
