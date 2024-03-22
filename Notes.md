# Development Notes

1. Architecture
    - The given architecture is different from the standard U-Net architecture
    in that there are two blocks with the same number of output channels
    in the downsample and upsample. This is not the standard U-Net architecture
    where each level of the downsample and upsample blocks have progressively
    increasing and decreasing number of channels.

    - Also there are 5 skip connections in place of the standard 4 skip connections in the standard U-Net architecture.

## Refs

1. Pytorch-UNET Repo: "git@github.com:milesial/Pytorch-UNet.git"
