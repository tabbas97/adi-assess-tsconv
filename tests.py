import pytest
import torch

def get_valid_model_baseline():
    from baseline_unet import UnetBaseline
    model = UnetBaseline()
    return model

def test_example():
    assert 1 == 1

def test_model_baseline_construct():
    model = get_valid_model_baseline()
    assert model is not None
    
def test_model_baseline_forward():
    model = get_valid_model_baseline()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape == x.shape

def test_UnetBaseline_warning_hw_insufficient_input_size():
    model = get_valid_model_baseline()
    import torch
    x = torch.randn(1, 1, 32, 256)
    with pytest.warns(UserWarning):
        y = model(x)
        
def test_UNetBaseline_Basic_Convergence():
    # Sanity check to see if the model can overfit on a small random noise dataset
    # If the model does not converge, then there is an issue with the model
    # likely due to the architecture not connecting properly
    model = get_valid_model_baseline()
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    torch.manual_seed(0)
    
    # Create a random noise dataset
    class RandomNoiseDataset(torch.utils.data.Dataset):
        def __init__(self, size, shape):
            self.size = size
            self.shape = shape
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return torch.randn(self.shape)
        
    # Create a random noise dataloader
    dataset = RandomNoiseDataset(100, (1, 256, 128))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Create a random noise target
    target = torch.randn(10, 1, 256, 128)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    list_of_losses = []
    
    # Train the model
    for epoch in tqdm(range(10)):
        loss_per_epoch = 0
        for i, data in enumerate(dataloader, 0):
            inputs = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            loss_per_epoch += loss.item()
            optimizer.step()
            
        list_of_losses.append(loss_per_epoch/len(dataloader))
            
    print("Losses : ", list_of_losses)
    # Check if the loss is decreasing
    assert np.mean(list_of_losses[:5]) > np.mean(list_of_losses[-5:])
    
@pytest.mark.parametrize("in_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("out_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("shift_axis", ["H", "W"])
@pytest.mark.parametrize("shift_len", [1, 2, 3, 4])
def test_shift_conv_construct(in_channels, out_channels, shift_axis, shift_len):
    from shift_conv import ShiftConv
    shift_conv = ShiftConv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = (3, 1),
        shift_axis = shift_axis,
        shift_len = shift_len
        )
    assert shift_conv is not None
    
@pytest.mark.parametrize("in_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("out_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("shift_axis", ["H", "W"])
@pytest.mark.parametrize("shift_len", [1, 2, 3, 4])
def test_shift_conv_forward(in_channels, out_channels, shift_axis, shift_len):
    from shift_conv import ShiftConv
    shift_conv = ShiftConv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = (3, 1),
        shift_axis = shift_axis,
        shift_len = shift_len
        )
    x = torch.rand(1, in_channels, 256, 256)
    y = shift_conv(x)
    assert y.shape[1] == out_channels
    assert y.shape[3] == x.shape[3]
    assert y.shape[2] == x.shape[2]
    
@pytest.mark.parametrize("in_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("out_channels", [4, 8, 16, 32])
def test_ts_conv_construct(in_channels, out_channels):
    from ts_conv import TSConv
    ts_conv = TSConv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = (3, 1)
        )
    assert ts_conv is not None
    
@pytest.mark.parametrize("in_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("out_channels", [4, 8, 16, 32])
def test_ts_conv_forward(in_channels, out_channels):
    from ts_conv import TSConv
    ts_conv = TSConv(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = (3, 1)
        )
    x = torch.rand(1, in_channels, 256, 256)
    y = ts_conv(x)
    assert y.shape[1] == out_channels
    assert y.shape[3] == x.shape[3]
    assert y.shape[2] == x.shape[2]