import torch
import torch.nn as nn

from typing import Union, Optional

class ShiftConv(nn.Module):
    """
    ShiftConv is a custom layer that performs a shifted convolution operation on the input tensor.
    The operation is defined as follows:
    1. The input tensor is split into three parts along the channel axis.
    2. The first part is left unchanged.
    3. The second part is rolled along the specified axis by the specified number of steps.
    4. The third part is rolled along the specified axis by the specified number of steps in the opposite direction.
    5. The three parts are concatenated back along the channel axis.
    6. The concatenated tensor is passed through a regular Conv2d layer.
    """
    
    shift_axes_map = {
        "W": 3,
        "H": 2
    }
    
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 kernel_size : Union[int, tuple], 
                 use_batchnorm : bool = False, 
                 shift_len : int = 1, 
                 shift_axis : str = "W"):
        """Initializes the ShiftConv layer

        Args:
            in_channels (int): Number of input channels to the conv layer
            out_channels (int): Number of output channels from the conv layer
            kernel_size (Union[int, tuple]): Size of the kernel. Only kx1 kernels are supported.
            use_batchnorm (bool, optional): Enables batchnorm operation. Defaults to False.
            shift_len (int, optional): The number of rows/cols to shift. Defaults to 1.
            shift_axis (str, optional): Selection of shift along height or width axis. Defaults to "W".

        Raises:
            ValueError: If shift_axis is not "W" or "H"
            ValueError: If shift_len is less than 1
            ValueError: If kernel_size is not an integer or a tuple of length 2
            ValueError: If kernel_size is a tuple of length 2 and the second element is not 1
        """

        super().__init__()
        
        if shift_axis not in ShiftConv.shift_axes_map:
            raise ValueError("Only shift along width or height is supported.")
        
        if shift_len < 1:
            raise ValueError("Shift length should be at least '1'. '0' shift is equivalent to a regular Conv2d")
        
        if not isinstance(kernel_size, int) and not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
            raise ValueError("Only kx1 kernels are supported for shift conv.")
        
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2 and kernel_size[-1] != 1:
           raise ValueError("Only kx1 kernels are supported for shift conv.")
            
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, 1)
        
        self.shift_len = shift_len
        
        self.shift_axis = ShiftConv.shift_axes_map[shift_axis]
        
        # print("out_channels", out_channels)
        # print("in_channels", in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size = kernel_size, 
            # stride = kernel_size,
            padding = 'same',
            )
        
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # Split the input tensor into three parts along the channel axis 
        # first half, third quarter, and last quarter
        static, dynamic_shift_neg, dynamic_shift_pos = (
                x[:, :x.shape[1]//2], 
                x[:, x.shape[1]//2:x.shape[1]//2 + x.shape[1]//4], 
                x[:, x.shape[1]//2 + x.shape[1]//4:]
                )
        
        # print(static.shape, dynamic_shift_neg.shape, dynamic_shift_pos.shape)
        
        # Roll the dynamic_shift tensors along the specified axis and set zero where the 
        # padding was to be done
        # Has the same effect as specified in the objective doc
        if self.shift_axis == 3:
            # print("Shift axis is 3")
            dynamic_shift_neg = torch.roll(
                dynamic_shift_neg, 
                shifts=-self.shift_len, 
                dims=self.shift_axis
                )
            dynamic_shift_neg[:, :, :, -self.shift_len:] = 0

            dynamic_shift_pos = torch.roll(
                dynamic_shift_pos, 
                shifts=self.shift_len, 
                dims=self.shift_axis
                )
            dynamic_shift_pos[:, :, :, :self.shift_len] = 0

        
        elif self.shift_axis == 2:
            dynamic_shift_neg = torch.roll(
                dynamic_shift_neg, 
                shifts=-self.shift_len, 
                dims=self.shift_axis
                )
            dynamic_shift_neg[:, :, -self.shift_len:] = 0
            
            dynamic_shift_pos = torch.roll(
                dynamic_shift_pos, 
                shifts=self.shift_len, 
                dims=self.shift_axis
                )
            dynamic_shift_pos[:, :, :self.shift_len] = 0
        
        # Concatenate the three tensors back along the channel axis
        x = torch.cat([static, dynamic_shift_neg, dynamic_shift_pos], dim=1)
        
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return x
    
    
if __name__ == "__main__":
    torch.manual_seed(5)
    
    shift_conv = ShiftConv(
        in_channels = 8,
        out_channels = 4,
        kernel_size = (3, 1),
        shift_axis = "H",
        shift_len = 2
        )    
    
    inp = torch.rand(1, 
                     shift_conv.conv.in_channels, 
                     256, 
                     384
                     )
    
    out = shift_conv(inp)
    
    # torch.onnx.export(
    #     shift_conv, 
    #     inp, 
    #     "shift_conv.onnx", 
    #     export_params=True, 
    #     opset_version=11, 
    #     do_constant_folding=True, 
    #     input_names = ["input"], 
    #     output_names = ["output"]
    #     )
    
    # Reduce precision for visual comparison
    out = out.detach().numpy().round(3)
    print(out)
    print("ShiftConv test passed!")