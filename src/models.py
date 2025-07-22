import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

import timm, torchray
import torchray.benchmark


class ModelEnv:

    def __init__(self, arch):
        self.arch = arch
        self.device = self.get_device()
        self.model = self.load_model(self.arch, self.device)
        self.shape = (224,224)

    def load_model(self, arch, dev):
        if arch.startswith('voc_'):
            model_arch = arch.replace('voc_','')
            model = torchray.benchmark.models.get_model(arch=model_arch, dataset="voc", convert_to_fully_convolutional=False)

        elif arch == 'vgg16NT':
            model = torchvision.models.vgg16(pretrained=False)
        elif arch == 'vgg16RT':            
            model = torchvision.models.vgg16(pretrained=False)
            output_weights_path = 'models/vgg16_retrained_n.pth'
            model.load_state_dict(torch.load(output_weights_path))

        elif 'resnet' in arch or 'vgg' in arch:
        # Get a network pre-trained on ImageNet.
            model = torchvision.models.__dict__[arch](pretrained=True)
            #for param in model.parameters():
            #    param.requires_grad_(False)        
        elif 'vit' in arch or 'convnext' in arch or 'densenet' in arch:
            model = timm.create_model(arch, pretrained=True)
        else:
            assert False, "unexpected arch"
        
        model.eval()        
        model = model.to(dev)
        return model

    def narrow_model(self, catidx, with_softmax=False):
        if "voc" in self.arch:
            modules = (
                [self.model] + 
                ([nn.Sigmoid()] if with_softmax else []) +
                [SelectKthLogit(catidx)])
        else:
            modules = (
                [self.model] + 
                ([nn.Softmax(dim=1)] if with_softmax else []) +
                [SelectKthLogit(catidx)])

        return nn.Sequential(*modules)
        
    def get_cam_target_layer(self):
        if self.arch == 'resnet50':
            return self.model.layer4[-1]
            #return self.model.layer4
        
        elif self.arch == 'vgg16':
            return self.model.features[-1]

        elif self.arch == 'densenet201':
            return self.model.features[-1]

        elif self.arch == 'convnext_base':
            return self.model.stages[-1].blocks[-1]
                        
        raise Exception('Unexpected arch')
    
    def get_cex_conv_layer(self):
        
        if self.arch == 'resnet50':
            return self.model.layer4[-1].conv3

        elif self.arch == 'vgg16':
            return self.model.features[-3]

        elif self.arch == 'convnext_base':
            return self.model.stages[-1].blocks[-1].conv_dw

        raise Exception('Unexpected arch')

    def get_device(self, gpu=0):
        device = torch.device(
            f'cuda:{gpu}'
            if torch.cuda.is_available() and gpu is not None
            else 'cpu')
        return device

    def get_transform(self):    
        if "voc" in self.arch:
            #print("voc transform 3")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to (224, 224)
                transforms.ToTensor(),  # Convert to tensor (scales to [0,1])
                transforms.Lambda(lambda x: x * 255.0)  # Multiply by 255
            ])
        elif 'resnet' in self.arch or 'vgg' in self.arch or 'convnext' in self.arch or 'densenet' in self.arch:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.shape),
                torchvision.transforms.CenterCrop(self.shape),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
            ])
        elif 'vit' in self.arch:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            assert False, "unexpected arch"
        return transform

    def get_image_ext(self, path):
        img = Image.open(path)
        # Pre-process the image and convert into a tensor
        transform = self.get_transform()
        x = transform(img).unsqueeze(0)
        return img, x.to(self.device)

    def get_image(self, path):
        return self.get_image_ext(path)[1]
    

class SelectKthLogit(nn.Module):
    def __init__(self, k):
        super(SelectKthLogit, self).__init__()
        self.k = k        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss  = nn.CrossEntropyLoss()

    def forward(self, x):
        if type(self.k) == int:            
            values = torch.stack([x], dim=-1)
            result = values[:,self.k,:]            
        else:
            result = x[:, self.k]                            
        
        return result
