from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.nn import Module
from torch.utils.data import DataLoader
from model.models import Net
from model.torch_wrapper import TorchWrapper
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

            self.training_log['learning_rate'].append(self.get_lr())

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
                    image = sample['image']
                    label = sample['keypoints']
                    if phase == 'train':
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image)
                    loss = self.criterion(outputs, label.float())
                    epoch_loss += loss.detach().cpu().numpy()

                    # Batch Accuracy
                    batch_predicted_labels = outputs.detach().cpu().numpy().reshape(-1)
                    batch_true_labels = label.detach().cpu().numpy().reshape(-1)
                    epoch_predicted_labels.append(batch_predicted_labels)
                    epoch_ground_truth.append(batch_true_labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    self.training_log[phase]['batch_loss'].append(loss.detach().cpu().numpy())
                    accuracy = r2_score(batch_true_labels, batch_predicted_labels)
                    self.training_log[phase]['batch_accuracy'].append(accuracy)

                    log_string = f'Epoch {epoch}/{epochs}, Batch {i+1}/{len(data_loader[phase])}'
                    log_string += f', Current Mean Epoch Loss: {epoch_loss / (i + 1):.2f}'
                    current_accuracy = r2_score(np.concatenate(epoch_ground_truth),
                                                np.concatenate(epoch_predicted_labels))
                    log_string += f'|Accuracy: {current_accuracy:.2f}'
                    logger.info(log_string)

                # Log Epoch (accuracy and loss)
                accuracy = r2_score(np.concatenate(epoch_ground_truth), np.concatenate(epoch_predicted_labels))
                self.training_log[phase]['epoch_accuracy'].append(accuracy)
                self.training_log[phase]['epoch_loss'].append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

