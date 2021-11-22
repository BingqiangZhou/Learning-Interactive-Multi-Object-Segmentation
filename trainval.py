
import os
from random import shuffle
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from utils import EvaluationUtils, LabelUtils, DistanceUtils

class TrainValModel():
    def __init__(self, model, train_or_val, device, epochs, dataset_name,
                    logger, writer, exp_ckpt_dir, sample_point_function,
                    use_course_learing=False, skip_when_instance_more_than=8, 
                    dataloader=[],
                    loss_function=None, optimizer=None, scheduler=None):
        
        self.train_or_val = train_or_val
        if self.train_or_val == 'train':
            if loss_function is None or optimizer is None:
                raise Exception('loss function and optimizer is not given.')

        self.model = model
        self.device = device
        self.logger = logger
        self.writer = writer
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.exp_ckpt_dir = exp_ckpt_dir
        self.use_course_learing = use_course_learing
        self.skip_when_instance_more_than = skip_when_instance_more_than
        self.train_dataloader, self.val_dataloader = dataloader
        self.sample_point_function = sample_point_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def trainval(self):
        if self.train_or_val == 'train':
            max_mean_iou = 0
            for epoch in range(self.epochs):
                # train
                self.one_epoch(self.train_dataloader, 'train', epoch+1)
                # save model
                torch.save(self.model.state_dict(), os.path.join(self.exp_ckpt_dir, f'lastest_epoch.pkl'))

                if epoch % 5 == 0:
                    # val
                    mean_val_iou = self.one_epoch(self.val_dataloader, 'val', epoch+1)
                    # save model
                    if mean_val_iou > max_mean_iou:
                        max_mean_iou = mean_val_iou
                        self.writer.add_text(f'train/current_max_val_mean_iou', f'epoch {epoch+1}, val mean iou {mean_val_iou}')
                        torch.save(self.model.state_dict(), os.path.join(self.exp_ckpt_dir, f'best_mean_iou_epoch.pkl'))
                
                if self.scheduler is not None:
                    self.scheduler.step()
        else: # val
            self.one_epoch(self.val_dataloader, 'val')

    def one_epoch(self, dataloader, train_or_val, epoch=0):
        self.logger.info(f'begin {train_or_val}ing...')
        
        iou_with_bg_list = []
        iou_without_bg_list = []
        loss_list = []
        embedding_loss_list = []
        classifier_loss_list = []
        for step, data in enumerate(dataloader):
            image, label = data
            p = 0 if train_or_val == 'val' else 0.5
            label_for_embedding = LabelUtils.get_nums_label(self.dataset_name, label, 0, shuffle=False)
            label_for_classifier = LabelUtils.get_nums_label(self.dataset_name, label, p, shuffle=True)
            if len(torch.unique(label_for_classifier)) < 2:
                continue
            image = image.to(self.device)
            label_for_embedding = label_for_embedding.to(self.device)
            label_for_classifier = label_for_classifier.to(self.device)
            
            # whether random the number of sample points
            random_nums_point = False if train_or_val == 'val' else True
            ## random sampleing point
            # binary_interactive_map, num_points = self.sample_point_function(label, random_nums_point)
            binary_interactive_map, num_points = self.sample_point_function(label_for_classifier, random_nums_point)

            ## skip when no points are sampled
            if binary_interactive_map is None:
                continue

            _, nums_instance, _, _ = binary_interactive_map.shape
            
            # if train_or_val == 'train' and nums_instance > self.skip_when_instance_more_than:
            if train_or_val == 'train':
                ## skip when image include 'nums_instance'
                if nums_instance > self.skip_when_instance_more_than:
                    continue

                ## course learning
                if self.use_course_learing:
                    if nums_instance > (((epoch) // 10) + 1):
                        continue
            
            ## transfrom binary interactive map to distance map
            distance_map = DistanceUtils.get_euclidean_distance_map(binary_interactive_map)

            # forward
            if train_or_val == 'train':
                outputs, embedding, visual_attention_maps, margin = self.model(image, distance_map)
                self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    outputs, embedding, visual_attention_maps, margin = self.model(image, distance_map)
            
            classifier_loss, embedding_loss = self.loss_function(embedding, outputs, label_for_classifier.long(), label_for_embedding.long(),
                                                     margin=margin, return_embedding_loss=True)
            total_loss = classifier_loss + embedding_loss

            ## backward
            if train_or_val == 'train':
                total_loss.backward()
                self.optimizer.step()
                
            ## calcualate iou
            iou_without_bg = EvaluationUtils.calculate_iou(outputs, label_for_classifier, skip_background=True)
            iou_with_bg = EvaluationUtils.calculate_iou(outputs, label_for_classifier, skip_background=False)
            loss_list.append(total_loss.item())
            embedding_loss_list.append(embedding_loss.item())
            classifier_loss_list.append(classifier_loss.item())
            iou_without_bg_list.append(iou_without_bg.item())
            iou_with_bg_list.append(iou_with_bg.item())

            self.logger.info(f'''{train_or_val}: epoch {epoch}, step {step+1}, number of instance {nums_instance}, num_points {num_points},loss {total_loss.item()}, embedding loss {embedding_loss.item()}, seg loss {classifier_loss.item()}, iou {iou_without_bg.item()} iou with bg {iou_with_bg.item()}''')
                                
        mean_iou_with_bg = np.mean(iou_with_bg_list)
        mean_iou_without_bg = np.mean(iou_without_bg_list)
        mean_loss = np.mean(loss_list)
        mean_embedding_loss = np.mean(embedding_loss_list)
        mean_classifier_loss = np.mean(classifier_loss_list)
        self.writer.add_scalar(f'{train_or_val}/mean_loss', mean_loss, epoch)
        self.writer.add_scalar(f'{train_or_val}/classifier_loss', mean_classifier_loss, epoch)
        self.writer.add_scalar(f'{train_or_val}/embedding_loss', mean_embedding_loss, epoch)
        self.writer.add_scalar(f'{train_or_val}/mean_iou_with_bg', mean_iou_with_bg, epoch)
        self.writer.add_scalar(f'{train_or_val}/mean_iou_without_bg', mean_iou_without_bg, epoch)
        
        self.logger.info(f'epoch {epoch} {train_or_val}ing end, mean iou: {mean_iou_without_bg}, mean iou with bg: {mean_iou_with_bg}, mean embedding loss: {mean_embedding_loss}, mean_classifier_loss {mean_classifier_loss} mean loss:{mean_loss}')

        return mean_iou_without_bg
