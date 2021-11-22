import torch
import torch.nn as nn

# Loss parameters
# dv = 0
# dd = 2.5
# gamma = 0.005

class SISDLFEmbeddingLoss(nn.Module):
    '''
        Semantic Instance Segmentation with a Discriminative Loss Function
        https://arxiv.org/abs/1708.02551
    '''
    def __init__(self, dd=2.5, dv=0, skip_background=True, background_label=0):
        super(SISDLFEmbeddingLoss, self).__init__()
        self.dd = dd
        self.dv = dv
        # self.gamma = gamma
        self.skip_background = skip_background
        self.background_label = background_label
    
    def forward(self, embedding_space, label):
        '''
            embedding_space: [n, c, h, w]
            label: [n, 1, h, w]
        '''
        device = label.device
        loss = torch.tensor([0.], device=device)

        b = embedding_space.shape[0]

        for i in range(b):
            loss = loss + self.calc_embedding_loss(embedding_space[i], label[i][0])

        loss = loss / b
            
        return loss

    def calc_embedding_loss(self, embedding_space, label):
        '''
            embedding_space: [c, h, w]
            label: [h, w]
        '''
        device = label.device

        label = label.flatten() # (h, w) -> (h * w)
        embedding_space = embedding_space.permute((1, 2, 0)) # (c, h, w) -> (h, w, c)
        embedding_space = embedding_space.flatten(start_dim=0, end_dim=1) # (h * w, c)

        instances = torch.unique(label)
        # print(instances)

        means = []
        var_loss = torch.tensor([0.], device=device)
        dist_loss = torch.tensor([0.], device=device)

        if len(instances) == 0: # no instances
            return torch.tensor([0.], device=device)

        # calculate intra-cluster loss
        for instance in instances:
            
            # get instance mean and distances to mean of all points in an instance
            if self.skip_background and instance == self.background_label: # Ignore background
                continue

            # collect all feature vector of a certain instance
            locations = (label == instance).nonzero().squeeze().long()
            vectors = torch.index_select(embedding_space, dim=0, index=locations)
            
            mean = torch.mean(vectors, dim=0, keepdim=True) # (n,c) -> (1, c)
            # print('mean', mean)
            dist2mean = torch.norm(vectors - mean, dim=-1)  # (n, c) -> (n, 1)
            # print('dist2mean',dist2mean)

            var_loss = var_loss + torch.mean(dist2mean, dim=0)
            # print(var_loss)
            means.append(mean)

        # get inter-cluster loss - penalize close cluster centers
        means = torch.stack(means)
        # print('means', means, means.shape)
        num_clusters = means.shape[0]
        # print('num_clusters', num_clusters)
        # dd = dd / num_clusters
        if num_clusters > 1:
            for i in range(num_clusters):
                for j in range(i+1, num_clusters):
                    dist = torch.norm(means[i]-means[j])
                    if dist < self.dd*2:
                        dist_loss = dist_loss + torch.pow(2*self.dd - dist, 2)/(num_clusters-1)
        
        # regularization term
        # reg_loss = torch.sum(torch.norm(means, 2, 1))

        # total_loss = (var_loss + dist_loss + self.gamma*reg_loss) / num_clusters
        # print(var_loss.requires_grad, dist_loss)
        total_loss = (var_loss + dist_loss) / num_clusters
        return total_loss

# embedding = torch.arange(18, requires_grad=True, dtype=torch.float32).reshape((1, 2, 3, 3)).cuda()
# label = torch.randint(0, 3, (1, 1, 3, 3)).cuda()
# print(embedding, label, sep='\n')

# loss_function = SISDLFEmbeddingLoss(dd=2.5, dv=0)
# loss = loss_function(embedding, label)
# print(loss, loss.requires_grad)