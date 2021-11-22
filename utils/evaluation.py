

import torch
from sklearn import random_projection

class EvaluationUtils():

    @staticmethod
    def logits_to_one_hot(tensor, dim=1):
        '''
        Description: 
        Arguments: 
            tensor: [n, c, h, w], also support other dimension tensor.
            dim: which dim to one hot
        Returns: 
            one_hot_tensor : [n, c, h, w]
        '''
        
        device = tensor.device

        index = torch.argmax(tensor, dim=dim, keepdim=True) # (n, 1, h, w)
        one_hot_tensor = torch.zeros_like(tensor, device=device)
        one_hot_tensor.scatter_(dim=dim, index=index, value=1) # (n, c, h, w)

        return one_hot_tensor

    @staticmethod
    def calculate_iou(logits, labels, logits_to_one_hot=True, skip_background=True, op='mean'):
        '''
        Description: 
        Arguments: 
            logits: [n, c, h, w]
            label: [n, 1, h, w]
            ops: ['mean', 'sum'], return sum of iou or mean of iou.
        Returns: 
            image_iou: sum of iou or mean of iou, determined by 'ops'
        '''

        ops = ['mean', 'sum']
        assert op in ops, "op must in ['mean', 'sum'], default 'mean'."
        
        if logits_to_one_hot:
            logits = EvaluationUtils.logits_to_one_hot(logits)

        device = labels.device
        batch_size = labels.shape[0]
        label_ids = torch.unique(labels)
        nums_object = len(label_ids) - 1 if skip_background else len(label_ids)

        image_iou = 0
        for b in range(batch_size):
            for index, id in enumerate(label_ids):
                if skip_background and id == 0:
                    continue
                current_mask = logits[b, index, :, :]
                current_label = (labels[b, 0, :, :] == id).float()
                add = current_mask.float() + current_label.float()
                interaction = (add == 2).float()
                union = (add >= 1).float()
                iou = torch.sum(interaction) / torch.sum(union)
                image_iou += iou
        
        if op == 'mean':
            image_iou = image_iou / nums_object / batch_size
        return image_iou

    @staticmethod
    def random_project_embedding_to_three_channel(embedding):
        
        transformer = random_projection.GaussianRandomProjection(n_components=3)

        b, c, h, w = embedding.shape
        
        embedding = embedding.squeeze(dim=0) #(1, c, h, w) -> (c, h, w)
        # print('embedding', embedding.shape)
        embedding = embedding.permute(1, 2, 0) #  (c, h, w) -> (h, w, c)
        embedding = embedding.reshape(h * w, c) # (h, w, c) -> (h * w, c)

        device = embedding.device
        if str(device)[:4] == 'cuda':
            embedding = embedding.cpu()
        embedding = transformer.fit_transform(embedding.detach().numpy()) # (h * w, c) -> (h * w, 3)
        embedding = embedding.reshape(h, w, 3)  # (h * w, 3) -> (h, w, 3)
        
        embedding = torch.from_numpy(embedding).to(device)
        embedding = embedding.permute(2, 0, 1) #  (h, w, 3) -> (3, h, w)
        embedding.unsqueeze_(dim=0)  # (c, h, w) -> (1, c, h, w)

        return embedding