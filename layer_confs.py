# Store layer configurations and parameters for the model

# All the MaxPooling and Upsample2D are size (2, 1)
# which will also set the stride to (2, 1) by default
POOLING_KERNEL_SIZE = (2, 1)
UPSAMPLE_KERNEL_SIZE = (2, 1)

# All the Conv2D layers have kernel size (3, 3), stride (1, 1)
# Padding allowed
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1

