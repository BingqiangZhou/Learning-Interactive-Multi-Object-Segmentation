import torch.nn as nn

class MultiBranchModule(nn.Module):
    def __init__(self, main_branch, sub_branchs):
        super(MultiBranchModule, self).__init__()

        assert len(sub_branchs) > 0, 'no sub-branch'

        self.main_branch = main_branch
        self.sub_branchs = nn.ModuleList(sub_branchs)

    def forward(self, x):
        x = self.main_branch(x)
        result = []
        for branch in self.sub_branchs:
            result.append(branch(x))
        
        return result