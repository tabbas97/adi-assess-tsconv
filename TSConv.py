from shift_conv import ShiftConv

class TSConv(ShiftConv):
    def __init__(self, in_channels, out_channels, kernel_size, use_batchnorm=False, *args, **kwargs):
        super().__init__(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            use_batchnorm = use_batchnorm, 
            shift_len = 1, 
            shift_axis = "W",
            *args, **kwargs
            )
        
    def forward(self, x):
        return super().forward(x)
    
if __name__ == "__main__":
    import torch
    
    ts_conv = TSConv(
        in_channels = 8,
        out_channels = 4,
        kernel_size = (3, 1)
        )
    
    inp = torch.rand(
        1,ts_conv.conv.in_channels, 
        256, 384
        )
    
    out = ts_conv(inp).detach().numpy().round(3)
    
    print(out)
    print(out.shape)
    print("TSConv valid")