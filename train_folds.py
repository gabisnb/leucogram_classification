import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from resnet import ResNet, ResidualBlock
import os
from pathlib import Path
import random
from PIL import Image
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="Training and evaluating a ResNet model for image classification.")
parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs.", default=50)

def train_fold(dataloader, model, loss_fn, epoch, optimizer, file_name, version = "", training_loss=None):

    # size of the dataset
    size = len(dataloader.dataset)

    # model in the training process
    model.train()

    totalLoss = 0

    for batch, (X, y) in enumerate(dataloader):
        # transforms the inputs to the device format (CPU or GPU)
        X, y = X.to(device), y.to(device)

        # prediction
        pred = model(X)

        # function loss estimation
        loss = loss_fn(pred, y)
        totalLoss += loss

        #### Backpropagation

        # gradient clearing
        optimizer.zero_grad()

        # gradient estimation
        loss.backward()

        # optimization of the parameters
        optimizer.step()

    print(f"Epoch average loss: {totalLoss/len(dataloader):>7f}")
    if training_loss is not None:
        training_loss.append(totalLoss/len(dataloader))
    # save results to file
    try:
        with open('folds/logs/train_' + file_name + version +'.txt', 'a') as f:
            f.write(f"Training Error Epoch {epoch}: {totalLoss/len(dataloader):>7f} \n\n")
    except:
        print("Error writing to file")


def validate_fold(dataloader, class_names, model, loss_fn, batch_size, epoch, lr, optimizer, version = "", validation_loss=None, validation_accuracy=None):
    all_preds, all_labels = [], []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            # transforms the inputs to the device format (CPU or GPU)
            X, y = X.to(device), y.to(device)

            # prediction
            pred = model(X)

            # get the class with the highest probability as the prediction and append to the list
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # calculate loss
            val_loss += loss_fn(pred, y).item()
            
            # calculate accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # average loss and accuracy
    val_loss /= num_batches
    correct /= size
    
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    if validation_accuracy is not None:
        validation_accuracy.append(100*correct)
    if validation_loss is not None:
        validation_loss.append(val_loss)

    # save results to file
    try:
        with open('logs/val' + str(batch_size) + '_' + str(lr) + '_' + str(optimizer) + version + '.txt', 'a') as f:
            f.write(f"Validation Error Epoch {epoch}: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n\n")
    except:
        print("Error writing to file")


    #### Confusion Matrix
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # save confusion matrix as image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig('confusion_matrix/val/resnet_cm_' + str(batch_size) + '_' + str(lr) + '_' + str(parser.parse_args().epochs) + '_' + str(optimizer) + version + '_val.png')

def test_fold(dataloader, class_names, model, loss_fn, file_name, fold = 1, version = ""):

    # size of the dataset and number of batches
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # model in the testing process
    model.eval()

    test_loss, correct = 0, 0

    # initialize lists of predictions and labels for confusion matrix
    all_preds, all_labels = [], []

    # disable gradient computation
    with torch.no_grad():

        for X, y in dataloader:
            # transforms the inputs to the device format (CPU or GPU)
            X, y = X.to(device), y.to(device)

            # prediction
            pred = model(X)

            # get the class with the highest probability as the prediction and append to the list
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # calculate loss
            test_loss += loss_fn(pred, y).item()
            
            # calculate accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # average loss and accuracy
    test_loss /= num_batches
    accuracy = 100*correct/size
    
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Number of correct predictions: {correct} out of {size}")

    # save results to file
    try:
        with open('folds/logs/test_' + file_name + version + '.txt', 'w') as f:
            f.write(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    except:
        print("Error writing to file")

    #### Confusion Matrix
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # save confusion matrix as image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix Fold ' + str(fold+1))

    plt.savefig('folds/confusion_matrix/test/normalized/test_normalized_' + file_name + version + '.png')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Fold ' + str(fold+1))

    plt.savefig('folds/confusion_matrix/test/absolute/test_absolute_' + file_name + version + '.png')
    return all_preds, all_labels, accuracy

def plot_training_loss(training_losses, file_name, version=""):
    fig, folds = plt.subplots(2, 3, figsize=(20, 12))
    i = 0
    all_losses_plot = folds.flat[5]
    for training_loss in training_losses:
        fold_plot = folds.flat[i]
        i = i+1
        training_loss = [loss.cpu().item() for loss in training_loss]

        # plot training loss curve
        # fold_plot.figure(figsize=(10, 6))
        fold_plot.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss Fold' + str(i))
        fold_plot.set_xlabel('Epoch')
        fold_plot.set_ylabel('Loss')
        fold_plot.set_title('Training Loss Fold ' + str(i))
        fold_plot.grid(True)
        
        all_losses_plot.plot(range(1, len(training_loss) + 1), training_loss, label='Fold ' + str(i))

    box = all_losses_plot.get_position()
    all_losses_plot.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    all_losses_plot.set_xlabel('Epoch')
    all_losses_plot.set_ylabel('Loss')
    all_losses_plot.set_title('All Folds Training Loss Curve')
    all_losses_plot.grid(True)
    all_losses_plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.tight_layout(pad=2.0)
    plt.savefig('folds/results/training_loss_' + file_name + version + '.png')

def plot_complete_confusion_matrix(all_preds, all_labels, class_names, file_name, version=""):
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # save confusion matrix as image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Complete Confusion Matrix')

    plt.savefig('folds/confusion_matrix/test/complete/test_normalized_' + file_name + version + '.png')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Complete Confusion Matrix')

    plt.savefig('folds/confusion_matrix/test/complete/test_absolute_' + file_name + version + '.png')

def plot_accuracies(accuracies, file_name, version=""):
    folds_ticks = [i+1 for i in range(len(accuracies))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(1, len(accuracies) + 1), accuracies, marker='o', color='blue', s=100)
    ax.set_xticks(folds_ticks)
    ax.set_xlabel('Fold')
    ax.set_ylim(min(accuracies)-5, max(accuracies)+5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.grid(True)
    plt.savefig('folds/results/test_accuracy_' + file_name + "_alt" + version + '.png')

def load_model_weights(model, weight_path):
    state_dict = torch.load(weight_path, weights_only=True)
    model.load_state_dict(state_dict)
    # model.eval()
    return model

def __main__():
    data_path = Path("/home/gabriela/projetos/datasets/")
    image_path = data_path / "cut_bboxes_folds"
    num_classes = 5
    class_names = ['bastonete', 'eosinofilo', 'linfocito', 'monocito', 'neutrofilo']
    num_folds = os.listdir(image_path).__len__() - 1
    epochs = parser.parse_args().epochs
    batch_size = 16
    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    fold_paths = [image_path / f"fold_{i+1}" for i in range(num_folds)]
    datasets_folds = []
    transform = Compose([
        Resize((224,224)),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    training_losses = []
    all_fold_preds = []
    all_fold_labels = []
    all_fold_accuracies = []

    version = ""
    file_name_prefix = "resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd"
    weights_prefix = "folds/weights/" + file_name_prefix
    if os.path.exists(weights_prefix + "_1.pth"):
        for i in range(1, 100):
            if not os.path.exists(weights_prefix + "_1_v" + str(i) + ".pth"):
                version = "_v" + str(i)
                break

    for fold in range(num_folds):
        dataset = datasets.ImageFolder(root=fold_paths[fold], transform=transform, target_transform=None)
        datasets_folds.append(dataset)

    # train_dataset = ConcatDataset([datasets_folds[i] for i in range(num_folds) if i != 4])
    # test_dataset = datasets_folds[4]

    for fold in range(num_folds):
        train_dataset = ConcatDataset([datasets_folds[i] for i in range(num_folds) if i != fold])
        cristal_dataset = datasets.ImageFolder(root=image_path / "cristal", transform=transform, target_transform=None)
        test_dataset = ConcatDataset([datasets_folds[fold], cristal_dataset])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        file_name = file_name_prefix + "_" + str(fold+1)

        #! usado apenas em testes rápidos sem treinamento
        # resnet = models.resnet34().to(device)
        # resnet = load_model_weights(resnet, "folds/weights/" + file_name + ".pth")
        # fold_preds, fold_labels, fold_accuracy = test_fold(test_dataloader, class_names, resnet, loss_fn, file_name, fold=fold, version="_v1.1")
        
        # all_fold_preds.extend(fold_preds)
        # all_fold_labels.extend(fold_labels)
        # all_fold_accuracies.append(fold_accuracy)
        # continue

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
        optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

        start, end = 0, 0
        training_loss = []
        file_name = file_name_prefix + "_" + str(fold+1)

        for t in range(epochs):
            print(f"-------------------------------\nEpoch {t+1}")
            start = time.time()
            train_fold(train_dataloader, resnet, loss_fn, (t+1), optimizer, file_name=file_name, version=version, training_loss=training_loss)
            end = time.time()
            print(f"Epoch time: {end-start}")

        training_losses.append(training_loss)
        torch.save(resnet.state_dict(), "folds/weights/" + file_name + version + ".pth")

        start = time.time()
        fold_preds, fold_labels, fold_accuracy = test_fold(test_dataloader, class_names, resnet, loss_fn, file_name, version=version)
        end = time.time()
        print(f"Test time: {end-start}")
        all_fold_preds.extend(fold_preds)
        all_fold_labels.extend(fold_labels)
        all_fold_accuracies.append(fold_accuracy)

    plot_training_loss(training_losses, file_name_prefix, version=version)
    plot_complete_confusion_matrix(all_fold_preds, all_fold_labels, class_names, file_name_prefix, version=version)
    plot_accuracies(all_fold_accuracies, file_name_prefix, version=version)


if __name__ == "__main__":
    __main__()