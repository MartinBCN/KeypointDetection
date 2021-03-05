from datetime import datetime
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from src.model.models import Net
from src.model.torch_wrapper import TorchWrapper
import logging
logger = logging.getLogger()


class KeypointDetector(TorchWrapper):

    def __init__(self):
        super().__init__()
        self.model = Net()

    def evaluate(self, data_loader: DataLoader):

        predicted_labels = []
        ground_truth = []

        for i, (image, label) in enumerate(data_loader):

            with torch.no_grad():

                predicted_labels.append(self.model.predict(image).detach().cpu().numpy())
                ground_truth.append(label.detach().cpu().numpy())

        # Log accuracy
        predicted_labels = np.concatenate(predicted_labels)
        ground_truth = np.concatenate(ground_truth)
        accuracy = (ground_truth == predicted_labels).mean()
        return accuracy, predicted_labels, ground_truth

    def train(self, data_loader: dict, epochs: int, early_stop_epochs: int = 5):

        start = datetime.now()
        for epoch in range(epochs):
            logger.info(('=' * 125))
            logger.info(f'Epoch {epoch}')
            logger.info(('=' * 125))

            for phase in ['train', 'validation']:

                logger.info(('-' * 125))
                logger.info(f'Phase {phase}')
                logger.info(('-' * 125))

                if phase not in self.training_log.keys():
                    self.training_log[phase] = {'epoch_loss': [], 'batch_loss': []}

                epoch_loss = 0
                epoch_ground_truth = []
                epoch_predicted_labels = []

                for i, sample in enumerate(data_loader[phase]):
                    logger.info(f'Batch {i+1}/{len(data_loader[phase])}')
                    image = sample['image']
                    label = sample['keypoints']
                    if phase == 'train':
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image)
                    loss = self.criterion(outputs, label.float())
                    epoch_loss += loss.detach().cpu().numpy()

                    logger.info(f'Current Mean Epoch Loss: {epoch_loss / (i + 1)}')

                    # epoch_predicted_labels.append(self.model.predict(image).detach().cpu().numpy())
                    # epoch_ground_truth.append(label.detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    if 'batch_loss' in self.training_log[phase].keys():
                        self.training_log[phase]['batch_loss'].append(loss.detach().cpu().numpy())
                    else:
                        self.training_log[phase]['batch_loss'] = [loss.detach().cpu().numpy()]

                # Log accuracy
                # accuracy = (np.concatenate(epoch_ground_truth) == np.concatenate(epoch_predicted_labels)).mean()
                accuracy = 0
                if 'accuracy' in self.training_log[phase].keys():
                    self.training_log[phase]['accuracy'].append(accuracy)
                else:
                    self.training_log[phase]['accuracy'] = [accuracy]

                # Log Epoch Loss
                if 'epoch_loss' in self.training_log[phase].keys():
                    self.training_log[phase]['epoch_loss'].append(epoch_loss)
                else:
                    self.training_log[phase]['epoch_loss'] = [epoch_loss]
