
from yacs.config import CfgNode

class Config():
    def __init__(self, config_file_path=None):
        self.basic_cfg = self.default_configs()
        if config_file_path is not None:
            self.basic_cfg.merge_from_file(config_file_path)
    
    def get_configs(self):
        self.basic_cfg.freeze()
        return self.basic_cfg

    def default_configs(self):
        
        _C = CfgNode(new_allowed=True)
        
        ## basic config
        _C.BASIC = CfgNode()
        _C.BASIC.EXP_NAME = 'default_config'
        _C.BASIC.EXP_RESULT_FOLDER = './exp_results'
        
        _C.BASIC.USE_GPU = True
        _C.BASIC.GPU = 0
        _C.BASIC.TRAIN_OR_VAL = 'train'
        _C.BASIC.MODEL_FILE_PATH = ''
        
        _C.BASIC.EPOCHS = 100
        _C.BASIC.SKIP_WHEN_INSTANCE_MORE_THAN = 10
        _C.BASIC.USE_COURSE_LEARNING = False

        ## dataset config
        _C.DATASET = CfgNode()
        _C.DATASET.DATASET_NAME = 'VOC'
        _C.DATASET.DATASET_ROOT_FOLDER = '/raid/home/guiyan/datasets'
        _C.DATASET.TRAIN_TRANSFORM = [[], []] # [['transform method', 'params1', 'params2'], []]
        _C.DATASET.VAL_TRANSFORM = [[], []]
        _C.DATASET.BATCH_SIZE = 1
        _C.DATASET.NUM_WORKERS = 0
        _C.DATASET.PIN_MEMORY = True


        ## random sample point
        _C.RANDOM_SAMPLE = CfgNode()
        _C.RANDOM_SAMPLE.NUMS_POINT = 10
        _C.RANDOM_SAMPLE.SAMPLE_METHODS = ["random_sample", "adaptive_dstep", "adaptive_dmargin"]
        # _C.RANDOM_SAMPLE.SAMPLE_PROB = 0.5 # prob of strategy 1, prob for strategy 2 equal `1-p`
        _C.RANDOM_SAMPLE.DMARGIN_INTERVAL = [0.05, 0.5]
        _C.RANDOM_SAMPLE.DSTEP_INTERVAL = [0.5, 0.8]
        _C.RANDOM_SAMPLE.D_STEP = 10
        _C.RANDOM_SAMPLE.D_MARGIN = 5

        ## loss function config
        _C.LOSS = CfgNode()

        _C.LOSS.LAMBDA_EMBEDDING = 1
        _C.LOSS.EMBEDDING_LOSS = CfgNode()
        _C.LOSS.EMBEDDING_LOSS.PACKAGE = 'loss'
        _C.LOSS.EMBEDDING_LOSS.LOSS = 'SISDLFEmbeddingLoss'
        _C.LOSS.EMBEDDING_LOSS.PARAMETERS = [0.128, 0] # dd, dv
        
        _C.LOSS.LAMBDA_CLASSIFIER = 1
        _C.LOSS.CLASSIFIER_LOSS = CfgNode()
        _C.LOSS.CLASSIFIER_LOSS.PACKAGE = 'torch.nn'
        _C.LOSS.CLASSIFIER_LOSS.LOSS = 'CrossEntropyLoss'
        _C.LOSS.CLASSIFIER_LOSS.PARAMETERS = [] # 

        ## optimizer
        _C.OPTIMIZER = CfgNode()
        _C.OPTIMIZER.OPTIMIZER_NAME = 'Adam'
        _C.OPTIMIZER.OPTIMIZER_PARAMS = [['net.params1', 1e-5, 'net.params2', 1e-3], 'other params such as weight_decay']
        _C.OPTIMIZER.USE_LR_SCHEDULER = True  
        _C.OPTIMIZER.LR_SCHEDULER = 'StepLR'
        _C.OPTIMIZER.LR_SCHEDULER_PARAMS = []

        ## network config
        _C.NETWORK = CfgNode()

        ## network.backbone
        _C.NETWORK.BACKBONE = CfgNode()
        _C.NETWORK.BACKBONE.BACKBONE_NAME = 'deeplabv3_resnet101' # 'fcn_resnet101'
        _C.NETWORK.BACKBONE.PRETRAINED = True
        _C.NETWORK.BACKBONE.REPLACE_BN_LAYER_TO_GN = False
        _C.NETWORK.BACKBONE.NUMS_CHANNEL_PER_GROUP_NORM = 32

        ## network.embedding
        _C.NETWORK.EMBEDDING = CfgNode()
        _C.NETWORK.EMBEDDING.WITH_MARGIN = False
        _C.NETWORK.EMBEDDING.CONV_TYPE = 'DSConv'
        _C.NETWORK.EMBEDDING.NORM_TYPE = 'GN'
        _C.NETWORK.EMBEDDING.ACTIVATION_TYPE = 'ReLU'
        _C.NETWORK.EMBEDDING.IN_CHANNELS = [256, 128]
        _C.NETWORK.EMBEDDING.OUT_CHANNEL = 128
        _C.NETWORK.EMBEDDING.NUMS_CHANNEL_PER_GROUP_NORM = 32
        _C.NETWORK.EMBEDDING.EXTRA_1X1_CONV_FOR_LAST_LAYER = False

        ## network.visual_attention
        _C.NETWORK.VISUAL_ATTENTION = CfgNode()
        _C.NETWORK.VISUAL_ATTENTION.CHANNEL_OF_PER_INTERACTIVATE_MAP = 1

        ## network.visual_attention.rnn_cells
        _C.NETWORK.VISUAL_ATTENTION.RNN_CELLS = CfgNode()
        _C.NETWORK.VISUAL_ATTENTION.RNN_CELLS.IN_CHANNEL = 257  # 256+1
        _C.NETWORK.VISUAL_ATTENTION.RNN_CELLS.HIDDEN_CHANNEL = 64
        _C.NETWORK.VISUAL_ATTENTION.RNN_CELLS.NUMS_CONV_LSTM_LAYER = 1
        _C.NETWORK.VISUAL_ATTENTION.RNN_CELLS.USE_BIAS = True

        ## network.visual_attention.foreground_convs
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS = CfgNode()
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.CONV_TYPE = 'DSConv'
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.NORM_TYPE = 'GN'
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.ACTIVATION_TYPE = 'ReLU'
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.IN_CHANNELS = [64, 128]
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.OUT_CHANNEL = 1
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.NUMS_CHANNEL_PER_GROUP_NORM = 32
        _C.NETWORK.VISUAL_ATTENTION.FOREGROUND_CONVS.EXTRA_1X1_CONV_FOR_LAST_LAYER = False

        ## network.visual_attention.background_convs
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS = CfgNode()
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.CONV_TYPE = 'DSConv'
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.NORM_TYPE = 'GN'
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.ACTIVATION_TYPE = 'ReLU'
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.IN_CHANNELS = [64, 64, 128]
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.OUT_CHANNEL = 1
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.NUMS_CHANNEL_PER_GROUP_NORM = 32
        _C.NETWORK.VISUAL_ATTENTION.BCAKGROUND_CONVS.EXTRA_1X1_CONV_FOR_LAST_LAYER = False

        ## network.dynamic_segmentation_head
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD = CfgNode()
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.CONV_TYPE = 'DSConv'
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.NORM_TYPE = 'GN'
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.ACTIVATION_TYPE = 'ReLU'
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.IN_CHANNELS = [128, 128]
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.OUT_CHANNEL = 1
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.NUMS_CHANNEL_PER_GROUP_NORM = 32
        _C.NETWORK.DYNAMIC_SEGMENTATION_HEAD.EXTRA_1X1_CONV_FOR_LAST_LAYER = False

        return _C.clone()