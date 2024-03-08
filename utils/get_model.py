from model import ResNet

def ResNet18(): return ResNet(18)
def ResNet34(): return ResNet(34)
def ResNet50(): return ResNet(50)
def ResNet101(): return ResNet(101)
def ResNet152(): return ResNet(152)

def get_model(resnet: str):
    resnets = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152
    }
    
    if resnet in resnets: 
        return resnets[resnet]()
    else:
        raise ValueError(f"Invalid ResNet: {resnet}")