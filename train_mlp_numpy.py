"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

# from sklearn import metrics as sklearn_metrics

import torch

import seaborn as sns
import matplotlib.pyplot as plt
from plot_conf_mat import plot_confusion_matrix

def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    # batch_size = targets.shape[0]
    # n_classes = predictions.shape[1]

    n_classes = 10 # hardcode 10 classes to solve the underdefined ill error
    
 
    one_hot_targets = np.eye(n_classes)[targets] 
    one_hot_predictions = np.eye(n_classes)[np.argmax(predictions, axis=1)] #converts loggits to one hot encoding

    
    # sk_conf_mat = sklearn_metrics.confusion_matrix(targets, np.argmax(predictions, axis=1))
    # sk_accuracy = metrics.accuracy_score(targets, label_predictions)
    # sk_precision_score = metrics.precision_score(targets, label_predictions, average=None)
    # sk_recall_score = metrics.recall_score(targets, label_predictions, average=None)
    # sk_f1_score = metrics.f1_score(targets, label_predictions, average=None)


    conf_mat = np.zeros((n_classes, n_classes))

    for expected_class in range(n_classes):
        for predicted_class in range(n_classes):
            if predicted_class == expected_class: # True positive
                conf_mat[expected_class][predicted_class]  = (np.sum(np.logical_and(one_hot_predictions[:, predicted_class]==1,\
                                    one_hot_targets[:, predicted_class]==1)))
            else: # False positive per class
                conf_mat[expected_class][predicted_class] = (np.sum(np.logical_and(one_hot_targets[:, expected_class] == 1, one_hot_predictions[:, predicted_class] == 1)))
    
    # assert np.allclose(conf_mat, sk_conf_mat)
    # return conf_mat, sk_accuracy, sk_precision_score, sk_recall_score, sk_f1_score
    return conf_mat

def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
        sk_accuracy,\
                                 sk_precision_score,      sk_recall_score, sk_f1_score,
    """
    n_classes = confusion_matrix.shape[1]
    metrics = {}
    metrics['accuracy'] = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    # metrics['precision'] = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    # metrics['recall'] = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    metrics['precision'] = np.array([confusion_matrix[class_][class_]/np.sum(confusion_matrix, axis=0)[class_] for class_ in range(n_classes)])
    metrics['recall'] = np.array([confusion_matrix[class_][class_]/np.sum(confusion_matrix, axis=1)[class_] for class_ in range(n_classes)])
    metrics['f1_beta'] = ((1 + beta**2) * (metrics['precision'] * metrics['recall'])) / (beta**2 * metrics['precision'] + metrics['recall'])
    # assert np.allclose(metrics['accuracy'], sk_accuracy)
    # assert np.allclose(metrics['precision'], sk_precision_score)
    # assert np.allclose(metrics['recall'], sk_recall_score)
    # assert np.allclose(metrics['f1_beta'], sk_f1_score)
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """
    test_iter = iter(data_loader)
    metrics = {}
    total_conf_mat = np.zeros((num_classes, num_classes))
    print('Start evaluating on the test set...')
    for _ in tqdm(range(len(data_loader))):
        images, labels = next(test_iter)
        images = images.reshape(images.shape[0], -1)
        predicitons = model.forward(images)
        conf_mat_per_batch = confusion_matrix(predicitons, labels)
        total_conf_mat += conf_mat_per_batch
    
    print(f'Confusion matrix: \n{total_conf_mat}\n')
    beta = 1
    metrics = confusion_matrix_to_metrics(total_conf_mat, beta=beta)
    print(f'Testing with f_{beta} score')
    plot_confusion_matrix(total_conf_mat, 'numpy')
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)    
    
    dataiter = iter(cifar10_loader['train'])
    images, labels = next(dataiter)
    n_classes = 10
    n_inputs = images.reshape(images.shape[0], -1).shape[1]
    #Initialize model and loss module
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()


    # Training loop including validation
    train_loss = {}
    train_loss_in_steps = {}
    val_loss = {}
    train_accuracies = {}
    val_accuracies = {}
    logging_info = {'train_loss': train_loss, 'train_loss_in_steps': train_loss_in_steps, 'val_loss': val_loss, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
    best_model = None
    best_val_accuracy = 0.
    for epoch in tqdm(range(epochs)):
        train_loss_ = 0.
        train_preds, count = 0., 0
        train_loss_per_batch = {}
        train_accuracy_per_batch = {}
        trainiter = iter(cifar10_loader['train'])
        for batch in range(len(cifar10_loader['train'])):
            images, labels = next(trainiter)
            images = images.reshape(images.shape[0], -1)
            
            model.clear_cache() # clear cache before each batch
            predicitons = model.forward(images)

            loss = loss_module.forward(predicitons, labels)
            dout = loss_module.backward(predicitons, labels)
            model.backward(dout)

            for layer in model.layers:
                if isinstance(layer, LinearModule):
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']

            train_preds += (predicitons.argmax(axis=1) == labels).sum()
            count += labels.shape[0]
            train_accuracy_per_batch[f'batch_{batch}'] = (predicitons.argmax(axis=1) == labels).sum() / labels.shape[0]
            
            train_loss_ += loss
            train_loss_per_batch[f'batch_{batch}'] = loss

            

            if batch % 10==0 and batch != 0:
                print(f"Epoch {epoch+1} - Batch {batch} - Train loss: {train_loss_per_batch[f'batch_{batch}']} - Train accuracy: {100*train_accuracy_per_batch[f'batch_{batch}']:.4f}%")
                train_loss_in_steps[f'epoch_{epoch+1}_batch_{batch}'] = train_loss_per_batch[f'batch_{batch}']

        train_acc = train_preds / count
        train_accuracies[f'epoch_{epoch+1}'] = train_acc
        train_loss[f'epoch_{epoch+1}'] = train_loss_ / len(cifar10_loader['train'])


        # Validation
        val_loss_ = 0.
        val_preds, count = 0., 0
        val_loss_per_batch = {}
        val_accuracy_per_batch = {}
        valiter = iter(cifar10_loader['validation'])
        for batch in range(len(cifar10_loader['validation'])):
            images, labels = next(valiter)
            images = images.reshape(images.shape[0], -1)
            
            predicitons = model.forward(images)
            loss = loss_module.forward(predicitons, labels)
            
            val_preds += (predicitons.argmax(axis=1) == labels).sum()
            count += labels.shape[0]
            val_accuracy_per_batch[f'batch_{batch}'] = (predicitons.argmax(axis=1) == labels).sum() / labels.shape[0]

            val_loss_ += loss
            val_loss_per_batch[f'batch_{batch}'] = loss
            
            if batch % 10 == 0:
                print(f"Epoch {epoch+1} In Batch {batch} - Val. loss of batch: {val_loss_per_batch[f'batch_{batch}']} - Val. Accuracy of batch : {100*val_accuracy_per_batch[f'batch_{batch}']:.4f}%")        
        
        val_acc = val_preds / count
        val_accuracies[f'epoch_{epoch+1}'] = val_acc
        val_loss[f'epoch_{epoch+1}'] = val_loss_ / len(cifar10_loader['validation'])
        
        if (len(val_accuracies) == 0) or (val_accuracies[f'epoch_{epoch+1}'] > best_val_accuracy):
            best_val_accuracy = val_accuracies[f'epoch_{epoch+1}']
            model.clear_cache()
            best_model = deepcopy(model)
        print('--------------------------------------------------')
        print(f"Epoch {epoch+1} - Avg Train loss: {train_loss[f'epoch_{epoch+1}']} - Avg. Val. loss: {val_loss[f'epoch_{epoch+1}']}\n - Avg. Train accuracy: {100*train_accuracies[f'epoch_{epoch+1}']:.4f}% - Avg. Val. Accuracy: {100*val_accuracies[f'epoch_{epoch+1}']:.4f}%")
        print('--------------------------------------------------')            


    print('Finished training.')
    print(f"Best validation accuracy: {100*best_val_accuracy:.4f}% in {max(val_accuracies, key=val_accuracies.get)}")

    # Test
    metrics = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = metrics['accuracy']
    print(f'Metrics: {metrics}')
    print(f"Test accuracy achieved: {100*test_accuracy:.4f}%")

    return best_model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    def plot_loss(logging_info):
        
        sns.set_style('darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        steps = len(logging_info['train_loss_in_steps'])
        ax1.plot(range(1, steps+1), list(logging_info['train_loss_in_steps'].values()), label='Training loss')
        ax1.set_xlabel('Batch steps')
        ax1.set_ylabel('Loss in Batch')
        ax1.set_title('Training loss in Batch steps across different epochs')
        
        epochs = len(logging_info['val_loss'])
        ax2.plot(range(1, epochs+1), list(logging_info['train_loss'].values()), "-o", label='Training loss')
        ax2.plot(range(1, epochs+1), list(logging_info['val_loss'].values()), "-o", label='Validation loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Avg. Loss')
        ax2.set_title('Avg. Training vs Validation loss')

        plt.legend()
        plt.tight_layout()

        folder_path = os.path.join(os.getcwd(), 'plots_assignment1')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, 'loss_curves_numpy.png')):
            plt.savefig(os.path.join(folder_path, 'loss_curves_numpy.png'))

    def plot_accuracy(logging_info):
        
        sns.set_style('darkgrid')
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        epochs = len(logging_info['val_accuracies'])

        min_val = 100*min(logging_info['val_accuracies'].values())
        max_val = 100*max(logging_info['val_accuracies'].values())

        ax1.plot(range(1, epochs+1), [100*acc_ for acc_ in logging_info['train_accuracies'].values()], "-o", label='Training accuracy')
        ax1.plot(range(1, epochs+1), [100*acc_ for acc_ in logging_info['val_accuracies'].values()], "-o", label='Validation accuracy')
        
        ax1.plot(np.argmin(list(logging_info['val_accuracies'].values())) + 1, min_val, "s", label=f"Min val. accuracy: {min_val:.4f}")
        ax1.plot(np.argmax(list(logging_info['val_accuracies'].values())) + 1, max_val, "D", label=f"Max val. accuracy: {max_val:.4f}")
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy %')
        ax1.set_title('Avg. Training vs Validation accuracy')

        plt.legend()
        plt.tight_layout()
        
        folder_path = os.path.join(os.getcwd(), 'plots_assignment1')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, 'acc_curves_numpy.png')):
            plt.savefig(os.path.join(folder_path, 'acc_curves_numpy.png'))

    plot_loss(logging_info)
    plot_accuracy(logging_info)
