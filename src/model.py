# pytorch models here
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels

def get_model(model_name='resnet18',pretrained=True):
    # model_name = 'alexnet'
    # model_name = 'resnet18'
    # model_name = "efficientnet_b0"
    # model_name = "mobilenet_v2"

    if model_name == "alexnet":
        if pretrained:
            model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[model_name](pretrained=None)
        # # print the model here to know whats going on.
        # for alexnet
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=1),
        )

    elif model_name == "resnet18":
        if pretrained:
            model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[model_name](pretrained=None)
        # # print the model here to know whats going on.
        # for resnet18
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=1),
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        # # print the model here to know whats going on.
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(model.classifier.in_features),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=model.classifier.in_features, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=1),
        )

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        # # print the model here to know whats going on.
        # for mobilenet
        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False
        
        in_feat_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_feat_dim),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=in_feat_dim, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1),
        )

    return model


if __name__ == '__main__':
    # model_names = ['alexnet', 'alexnet', 'resnet18', "efficientnet_b0", "mobilenet_v2"]
    # for model_name in model_names:
    #     model = get_model(model_name=model_name,pretrained=None)
    #     num_params = sum(p.numel() for p in model.parameters())
    #     print(f"{model_name} total parameters: {num_params/10**6} Million")


    # model_name = 'mobilenet_v2'
    model_name = 'resnet18'
    model = get_model(model_name=model_name)
    print(model)
    # for param in model.parameters():
        # print(param.shape, param.requires_grad)