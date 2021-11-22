import torch
import torchvision

class Backbone(torch.nn.Module):
    '''
        input: [b, 3, h, w]
        output: [b, 512, h, w] or [b, 256, h, w]
    '''
    def __init__(self, backbone_name='deeplabv3_resnet101', pretrained=True, 
                    replace_bn_to_gn=False, nums_norm_channel_per_group=32):
        super(Backbone, self).__init__()
        
        # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
        # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
        # 'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
        # 'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
        #  ~/.cache/torch/checkpoints/

        # 'deeplabv3_resnet50' and 'fcn_resnet50' don't support pretrained any more.
        self.support_backbone_list = ['deeplabv3_resnet101', 'fcn_resnet101', 'deeplabv3_resnet50', 'fcn_resnet50']

        assert backbone_name in self.support_backbone_list, \
            f"the backbone current only support {self.support_backbone_list}"

        self.backbone_name = backbone_name
        self.nums_norm_channel_per_group = nums_norm_channel_per_group
        
        self.backbone_net = getattr(torchvision.models.segmentation, self.backbone_name)(pretrained=pretrained)
        
        ## 3 for deeplabv3, 4 for fcn, becuase 'fcn' have an extra dorpout layer
        # remove_classifier_last_n_layer = 3 if self.backbone_name == 'deeplabv3_resnet101' else 4
        remove_classifier_last_n_layer = 1
        self.backbone_net.classifier = torch.nn.Sequential(*list(self.backbone_net.classifier.children())[:-remove_classifier_last_n_layer])
        self.backbone_net.aux_classifier = torch.nn.Sequential()
        
        if replace_bn_to_gn:
            self.replace_bn_to_gn_layer()

    def get_out_channel(self):
        out_channel = 256 if self.backbone_name == 'deeplabv3_resnet101' else 512
        return out_channel

    def forward(self, x):
        out = self.backbone_net(x)
        return out['out'] 

    def replace_bn_to_gn_layer(self):
        for name, layer in self.backbone_net.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                # print(name, layer, getattr(layer, 'num_features'))
                
                # locate BN module by name
                temp_model = self.backbone_net
                for attr_name in name.split('.')[:-1]:
                    temp_model = getattr(temp_model, attr_name)
                
                # get GN's input channel
                in_channel = getattr(layer, 'num_features')

                # replace BN to GN
                gn_layer = torch.nn.GroupNorm(in_channel//self.nums_norm_channel_per_group, in_channel)
                setattr(temp_model, name.split('.')[-1], gn_layer)