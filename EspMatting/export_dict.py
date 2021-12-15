import torch
from model import segnet
myModel = torch.load("./pre_trained/attunet/model/model_obj.pth")
print(myModel.state_dict())

torch.save(myModel.state_dict(), './esp_dict.pth')


