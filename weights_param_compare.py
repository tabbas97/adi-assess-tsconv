from UNet import Unet

import matplotlib.pyplot as plt

baseline_model = Unet(conv = Unet.convOptions.get("vanilla"))
tsconv_model = Unet(conv = Unet.convOptions.get("TSConv"))

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_total_params(model):
    return sum(p.numel() for p in model.parameters())

def get_model_params(model):
    return {
        "trainable_params": get_trainable_params(model),
        "total_params": get_total_params(model)
    }
    
baseline_params = get_model_params(baseline_model)
tsconv_params = get_model_params(tsconv_model)

print("UnetTSConv trainable params : ", tsconv_params["trainable_params"])
print("UnetTSConv total params : ", tsconv_params["total_params"])

print("UnetBaseline trainable params : ", baseline_params["trainable_params"])
print("UnetBaseline total params : ", baseline_params["total_params"])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(["UnetTSConv", "UnetBaseline"], [tsconv_params["trainable_params"], baseline_params["trainable_params"]])
# Legend
# ax.legend()
plt.xlabel("Model")
plt.ylabel("Trainable Parameters")
plt.show()
