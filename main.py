

import torch
import argparse

from config import Config
from trainval import TrainValModel
from builder import Builder


#### 1. initilize
#########################################################################################################################

## load configuration from file
parser = argparse.ArgumentParser(description='train or val for our model')
parser.add_argument('-c', '--config_file_path', type=str, default="./config/default_config.yaml", help='the path of yaml configuration file')
args = parser.parse_args()

config = Config(args.config_file_path).get_configs()
basic_config = config.get('BASIC')
network_configs = config.get('NETWORK')

## logger and summary writer
logger, writer, exp_ckpt_dir = Builder.setup_logger_and_summary_writer(basic_config)
logger.info(config)

## 
device = torch.device(f"cuda:{basic_config.get('GPU')}" if basic_config.get('USE_GPU') else 'cpu')
model = Builder.build_model(basic_config, network_configs, device)
model.eval()

## loss function
loss_function = Builder.build_loss_function(config.get('LOSS'))
# print(loss_function.__str__())

## optimizer
optimizer, scheduler = Builder.build_optimizer(config.get('OPTIMIZER'), model)
# print(optimizer.__str__())

## 2. begin train or val
#########################################################################################################################

# load dataset
# print(config.get("DATASET"))
train_dataloader, val_dataloader = Builder.build_dataset_dataloader(config.get("DATASET"))
# print(len(train_dataloader), len(val_dataloader))

# setup random sample point strategy
sample_point_function = Builder.setup_sample_point_function(config.get("RANDOM_SAMPLE"))

TrainValModel(model, basic_config.get('TRAIN_OR_VAL'), device,
                basic_config.get('EPOCHS'),
                config.get('DATASET').get('DATASET_NAME'),
                logger, writer, exp_ckpt_dir, sample_point_function,
                use_course_learing=basic_config.get('USE_COURSE_LEARNING'), 
                skip_when_instance_more_than=basic_config.get('SKIP_WHEN_INSTANCE_MORE_THAN'), 
                dataloader=[train_dataloader, val_dataloader],
                loss_function=loss_function, optimizer=optimizer, scheduler=scheduler).trainval()