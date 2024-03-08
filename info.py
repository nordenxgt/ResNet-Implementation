from torchinfo import summary

from utils import get_model

def main():
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    for resnet in resnets:
        model = get_model(resnet)
        print(resnet)
        summary(model, input_size=[1, 3, 224, 224])
        print("\n\n")

if __name__ == "__main__":
    main()