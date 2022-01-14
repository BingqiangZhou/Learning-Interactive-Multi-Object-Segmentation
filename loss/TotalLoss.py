import torch
import torch.nn as nn

class TotalLoss(nn.Module):
    def __init__(self, embedding_loss, classifier_loss, lambda_embedding=1, lambda_classifier=1):
        super(TotalLoss, self).__init__()

        self.embedding_loss = embedding_loss
        self.classifier_loss = classifier_loss
        
        self.lambda_embedding = lambda_embedding if lambda_embedding > 0 else 0
        self.lambda_classifier = lambda_classifier if lambda_classifier > 0 else 0

    def forward(self, embedding_space, outputs, target, embedding_target, margin=None, 
                    return_embedding_loss=False):

        device = target.device
        _, n, _, _ = outputs.shape

        if margin is not None:
            embedding_loss = self.embedding_loss(embedding_space, margin, embedding_target)
        else:
            embedding_loss = self.embedding_loss(embedding_space, embedding_target)

        embedding_loss = self.lambda_embedding * embedding_loss
        classifier_loss = self.lambda_classifier * self.classifier_loss(outputs, target[:, 0, :, :]) / n
        # return classifier_loss, embedding_loss
        if self.lambda_embedding != 0:
            total_loss = embedding_loss + classifier_loss
        else:
            total_loss = classifier_loss
        # print(f'embedding_loss: {embedding_loss}, classifier_loss: {classifier_loss}, total_loss: {total_loss}')
        # print(loss.requires_grad)
        if return_embedding_loss:
            return total_loss, embedding_loss
        return total_loss
        


