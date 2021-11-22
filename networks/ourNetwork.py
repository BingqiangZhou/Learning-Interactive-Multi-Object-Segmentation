

import torch
import torch.nn as nn
from collections import Sequence

class OurNetwork(nn.Module):
    def __init__(self, backbone, embedding, visual_attention, dynamic_segmentation_head):
        super(OurNetwork, self).__init__()

        ## init backbone
        self.backbone = backbone

        ## init embedding
        self.embedding = embedding

        ## init visual attention
        self.visual_attention = visual_attention

        ## init dynamic segmentation head
        self.dynamic_segmentation_head = dynamic_segmentation_head

        # self.softmax = nn.Softmax2d()


    def forward(self, image, distance_map):
        # print(torch.unique(distance_map))
        feature_map = self.backbone(image)

        margin = None
        embedding_space = self.embedding(feature_map)
        if isinstance(embedding_space, Sequence):
            embedding_space, margin = embedding_space
        
        visual_attention_maps = self.visual_attention(feature_map, distance_map)

        segmentation_result = []
        for object_visual_attention in torch.split(visual_attention_maps, 1, dim=1):
            embedding_feature = object_visual_attention * embedding_space + embedding_space
            out = self.dynamic_segmentation_head(embedding_feature)
            segmentation_result.append(out)
        
        segmentation_result = torch.cat(segmentation_result, dim=1)
        
        # we return embedding to calculate loss
        return segmentation_result, embedding_space, visual_attention_maps, margin