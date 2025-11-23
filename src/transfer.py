import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=2):
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True)

    else:
        raise ValueError(f"Model {model_name} is not known")

    # Freeze all parameters in the model

    for param in model_transfer.parameters():
        param.requires_grad = False

    # get numbers of features extracted by the backbone
    num_ftrs = model_transfer.fc.in_features

    # linear layer
    model_transfer.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, n_classes),
    )

    return model_transfer
