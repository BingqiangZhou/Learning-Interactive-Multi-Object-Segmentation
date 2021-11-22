import os
import sys
import torch
import importlib
import torchvision
from datetime import datetime
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks import Backbone, VisualAttention, OurNetwork
from networks.base import StackConv2D, StackConvLSTMCell, MultiBranchModule
from datasets import Datasets, Transfroms
from utils import RandomSamplePointUtils
from loss import TotalLoss

class Builder():
    @staticmethod
    def build_model(basic_config, network_configs, device=None):
        ## backbone
        backbone = Builder.build_backbone(network_configs.get('BACKBONE'))

        ## embedding
        embedding = Builder.build_embedding_head(network_configs.get('EMBEDDING'))

        ## visual attention
        visual_attention = Builder.build_visual_attention(network_configs.get('VISUAL_ATTENTION'))

        ## dynamic segmentation head
        dynamic_segmentation_head = Builder.build_stack_convs(network_configs.get('DYNAMIC_SEGMENTATION_HEAD'))
        
        if device is None:
            device = torch.device("cpu" if basic_config.get('USE_GPU') else f"cuda:{basic_config.get('GPU')}")
        
        model = OurNetwork(backbone, embedding, visual_attention, dynamic_segmentation_head)

        ## load model parameters
        model_file_path = basic_config.get('MODEL_FILE_PATH')
        if model_file_path != '' and os.path.exists(model_file_path):
            model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.to(device)

        return model

    @staticmethod
    def build_backbone(backbone_config):
        return Backbone(backbone_name=backbone_config.get('BACKBONE_NAME'), 
                    pretrained=backbone_config.get('PRETRAINED'),
                    replace_bn_to_gn=backbone_config.get('REPLACE_BN_LAYER_TO_GN'),
                    nums_norm_channel_per_group=backbone_config.get('NUMS_CHANNEL_PER_GROUP_NORM'))
    
    @staticmethod
    def build_visual_attention(visual_attention_config):
        rnn_cells_config = visual_attention_config.get('RNN_CELLS')
        rnn_cells = StackConvLSTMCell(input_dim=rnn_cells_config.get('IN_CHANNEL'), 
                                        hidden_dim=rnn_cells_config.get('HIDDEN_CHANNEL'), 
                                        use_bias=rnn_cells_config.get('USE_BIAS'), 
                                        nums_conv_lstm_layer=rnn_cells_config.get('NUMS_CONV_LSTM_LAYER'))
        
        foreground_convs_config = visual_attention_config.get('FOREGROUND_CONVS')
        foreground_convs = Builder.build_stack_convs(foreground_convs_config)

        background_convs_config = visual_attention_config.get('BCAKGROUND_CONVS')
        background_convs = Builder.build_stack_convs(background_convs_config)

        return VisualAttention(rnn_cells, foreground_convs, background_convs,
                                channel_of_per_interactive_map=visual_attention_config.get('CHANNEL_OF_PER_INTERACTIVATE_MAP'))

    @staticmethod
    def build_stack_convs(stack_convs_config):
        return StackConv2D(conv_type=stack_convs_config.get('CONV_TYPE'),
                            norm_type=stack_convs_config.get('NORM_TYPE'),
                            activation_type=stack_convs_config.get('ACTIVATION_TYPE'),
                            in_channels=stack_convs_config.get('IN_CHANNELS'),
                            out_channel=stack_convs_config.get('OUT_CHANNEL'),
                            nums_norm_channels_per_groups=stack_convs_config.get('NUMS_CHANNEL_PER_GROUP_NORM'),
                            extra_1x1_conv_for_last_layer=stack_convs_config.get('EXTRA_1X1_CONV_FOR_LAST_LAYER'))
    
    @staticmethod
    def build_embedding_head(embedding_config):
        if not embedding_config.get('WITH_MARGIN'): # without margin
            return Builder.build_stack_convs(embedding_config)
        else: # with margin
            main_branch = StackConv2D(conv_type=embedding_config.get('CONV_TYPE'),
                            norm_type=embedding_config.get('NORM_TYPE'),
                            activation_type=embedding_config.get('ACTIVATION_TYPE'),
                            in_channels=embedding_config.get('IN_CHANNELS')[:-1],
                            out_channel=embedding_config.get('IN_CHANNELS')[-1],
                            nums_norm_channels_per_groups=embedding_config.get('NUMS_CHANNEL_PER_GROUP_NORM'),
                            extra_1x1_conv_for_last_layer=embedding_config.get('EXTRA_1X1_CONV_FOR_LAST_LAYER'))
            embedding_branch = StackConv2D(conv_type=embedding_config.get('CONV_TYPE'),
                            in_channels=[embedding_config.get('IN_CHANNELS')[-1]],
                            out_channel=embedding_config.get('OUT_CHANNEL'))
            margin_branch = StackConv2D(conv_type=embedding_config.get('CONV_TYPE'),
                            in_channels=[embedding_config.get('IN_CHANNELS')[-1]],
                            out_channel=1)
            return MultiBranchModule(main_branch, [embedding_branch, margin_branch])

    # @staticmethod
    # def build_loss_function(loss_config):
    #     loss_init_params = loss_config.get('INIT_PARAMETERS')
    #     loss_package = importlib.import_module(loss_config.get('PACKAGE'))
    #     loss_class = getattr(loss_package, loss_config.get('LOSS_FUNCTION'))
    #     if len(loss_init_params) != 0:
    #         loss_function = loss_class(*loss_init_params)
    #     else:
    #         loss_function = loss_class()
        
    #     return loss_function        

    @staticmethod
    def build_loss_function(loss_config):
        def build_loss(loss_config):
            loss_init_params = loss_config.get('PARAMETERS')
            loss_package = importlib.import_module(loss_config.get('PACKAGE'))
            loss_class = getattr(loss_package, loss_config.get('LOSS'))
            if len(loss_init_params) != 0:
                loss_function = loss_class(*loss_init_params)
            else:
                loss_function = loss_class()
            return loss_function
        
        embedding_loss_function = build_loss(loss_config.get('EMBEDDING_LOSS'))
        classifier_loss_function = build_loss(loss_config.get('CLASSIFIER_LOSS'))

        loss_function = TotalLoss(embedding_loss_function, classifier_loss_function, 
                                    loss_config.get('LAMBDA_EMBEDDING'), loss_config.get('LAMBDA_CLASSIFIER'))

        return loss_function

    @staticmethod
    def build_optimizer(optimizer_config, model):
        # OPTIMIZER_NAME: 'Adam'
        # # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
        # OPTIMIZER_PARAMS: [['backbone', 1e-6], 1e-4, [0.9, 0.999], 1e-08, 1e-5]

        params = optimizer_config.get('OPTIMIZER_PARAMS')
        module_param_names = params[0]
        # print(module_param_names)
        other_optim_params = params[1:]
        ids = []
        params = []
        i = 0
        while i < len(module_param_names):
            module_name_list = module_param_names[i].split('.')
            name = module_name_list[0]
            module = getattr(model, name)
            for name in module_name_list[1:]:
                module = getattr(module, name)
            ids.extend(list(map(lambda x: id(x), module.parameters())))
            params.append({"params": module.parameters(), "lr": module_param_names[i+1]})
            i += 2
        
        ## without given parameters
        parameters_not_be_given = list(filter(lambda x: id(x) not in ids, model.parameters()))
        params.append({"params": parameters_not_be_given})
        # optimizer = torch.optim.Adam([ {'params': model.backbone.parameters(), 'lr': config.get('BACKBONE_LEARNING_RATE')}, 
        #                                 {'params': parameters_not_in_backbone}], 
        #                                 lr=config.get('LEARNING_RATE'),
        #                                 weight_decay=config.get('WEIGHT_DECAY'))
        optimizer = getattr(torch.optim, optimizer_config.get('OPTIMIZER_NAME'))(params, *other_optim_params)
        scheduler = None
        # USE_LR_SCHEDULER: True  
        # LR_SCHEDULER: 'LambdaLR'
        # # lr_lambda, last_epoch=-1, verbose=False
        # LR_SCHEDULER_PARAMS: []
        # print(*optimizer_config.get("LR_SCHEDULER_PARAMS"))
        if optimizer_config.get("USE_LR_SCHEDULER"):
            scheduler = getattr(torch.optim.lr_scheduler, optimizer_config.get("LR_SCHEDULER"))(optimizer, *optimizer_config.get("LR_SCHEDULER_PARAMS"))
            
        return optimizer, scheduler
    
    @staticmethod
    def build_dataset_dataloader(dataset_config):
        
        # print(dataset_config)
        def build_transform(transform_config):
            ## [['transform method', 'params1', 'params2'], []]
            transform_list = []
            for transform_method in transform_config:
                # compatible with old `pytorch` version 
                transform_params =  [tuple(params) for params in transform_method[1:] if isinstance(params, list)]
                transform_list.append(getattr(torchvision.transforms, transform_method[0])(*transform_params))
            
            return Transfroms(transform_list)

        dataset = Datasets(dataset_config.get("DATASET_NAME"))
        train_dataset = dataset.get_dataset(root_dir=dataset_config.get("DATASET_ROOT_FOLDER"), 
                        target_type='Object', image_set='train',
                        transforms=build_transform(dataset_config.get("TRAIN_TRANSFORM")))

        val_dataset = dataset.get_dataset(root_dir=dataset_config.get("DATASET_ROOT_FOLDER"), 
                        target_type='Object', image_set='val',
                        transforms=build_transform(dataset_config.get("VAL_TRANSFORM")))
        
        train_dataloader = DataLoader(train_dataset, batch_size=dataset_config.get("BATCH_SIZE"), shuffle=True, 
                                        num_workers=dataset_config.get("NUM_WORKERS"),
                                        pin_memory=dataset_config.get("PIN_MEMORY"))
        val_dataloader = DataLoader(val_dataset, batch_size=dataset_config.get("BATCH_SIZE"), shuffle=False,
                                        num_workers=dataset_config.get("NUM_WORKERS"), 
                                        pin_memory=dataset_config.get("PIN_MEMORY"))

        return train_dataloader, val_dataloader
    
    def setup_logger_and_summary_writer(basic_config):

        def make_dir(dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        ## create the floder of saving expriment result
        make_dir(basic_config.get('EXP_RESULT_FOLDER'))

        ## create experiment folder
        exp_root_folder = os.path.join(basic_config.get('EXP_RESULT_FOLDER'), basic_config.get('EXP_NAME'))
        make_dir(exp_root_folder)
        
        train_val = basic_config.get('TRAIN_OR_VAL')
        exp_folder = os.path.join(exp_root_folder, train_val)
        make_dir(exp_folder)
        
        now_str = datetime.now().__str__().replace(' ', '_')
        
        ## create a logger folder and a logger file
        exp_log_folder = os.path.join(exp_folder, "log")
        make_dir(exp_log_folder)
        logger_path = os.path.join(exp_log_folder, now_str + ".log")
        logger.remove(0) # remove default handler: https://github.com/Delgan/loguru/issues/51
        fromat = '{time:YYYY-MM-DD at HH:mm:ss} - {level} - {file}:{line} - {message}'
        level='INFO'
        logger.add(logger_path, level=level, format=fromat)
        logger.add(sys.stdout, level=level, format=fromat)

        ## create a folder for saving the file of visualization
        exp_visual_dir = os.path.join(exp_folder, "visual")
        make_dir(exp_visual_dir)
        summary_path = os.path.join(exp_visual_dir, now_str)
        writer = SummaryWriter(summary_path)
        
        exp_ckpt_dir = None
        if train_val == 'train':
            exp_ckpt_dir = os.path.join(exp_root_folder, 'train', "checkpoints", now_str)
            make_dir(exp_ckpt_dir)
            
        return logger, writer, exp_ckpt_dir
    
    def setup_sample_point_function(sample_point_config):
        # print(sample_point_config.get('STRATEGY_1_MARGIN_INTERVAL'), 
        #                             sample_point_config.get('STRATEGY_2_MARGIN_INTERVAL'))
        # return lambda label: RandomSamplePointUtils.sample_points_by_random_strategy(
        #                             label, sample_point_config.get('NUMS_POINT'), 
        #                             sample_point_config.get('SAMPLE_PROB'),
        #                             sample_point_config.get('STRATEGY_1_MARGIN_INTERVAL'), 
        #                             sample_point_config.get('STRATEGY_2_MARGIN_INTERVAL'))
        # return lambda label: RandomSamplePointUtils.sample_points(
        #                         label, sample_point_config.get('NUMS_POINT'),
        #                         sample_point_config.get('D_STEP'),
        #                         sample_point_config.get('D_MARGIN'))
        return lambda label, random_nums_points: RandomSamplePointUtils.sample_points(label, sample_point_config.get('NUMS_POINT'), 
                        sample_method_list=sample_point_config.get('SAMPLE_METHODS'), 
                        d_step=sample_point_config.get('D_STEP'), d_margin=sample_point_config.get('D_MARGIN'),
                        dmargin_interval=sample_point_config.get('DMARGIN_INTERVAL'), 
                        dstep_interval=sample_point_config.get('DSTEP_INTERVAL'),
                        random_nums_points=random_nums_points)
        

    
