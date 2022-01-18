import torchvision
import torch

model = torchvision.models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
# script_model.save('s8_deployment/model_store/deployable_model.pt')