import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.transforms.functional import equalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# from resnet import ResNet, ResidualBlock
import os
from pathlib import Path
import random
# from PIL import Image
import time
import argparse
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="Training and evaluating a ResNet model for image classification.")
parser.add_argument("-i", "--starting-fold", type=int, help="Starting fold id for training.", default=1)
parser.add_argument("-f", "--ending-fold", type=int, help="Ending fold id for training.", default=5)

def train_fold(dataloader, model, loss_fn, epoch, optimizer, file_name="resnet", experiment_path="resnet/v0.0", training_loss=None):

    # size of the dataset
    size = len(dataloader.dataset)

    # model in the training process
    model.train()

    totalLoss = 0
    
    try:
        if not os.path.exists('folds/' + experiment_path + '/logs/'):
            os.makedirs('folds/' + experiment_path + '/logs/')
        open('folds/' + experiment_path + '/logs/train_' + file_name +'.txt', 'w').close()
    except:
        print("Error writing to file")

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
        if not os.path.exists('folds/' + experiment_path + '/logs/'):
            os.makedirs('folds/' + experiment_path + '/logs/')
        with open('folds/' + experiment_path + '/logs/train_' + file_name +'.txt', 'a') as f:
            f.write(f"{epoch}:{totalLoss/len(dataloader):>7f}\n")
    except:
        print("Error writing to file")
    return (totalLoss/len(dataloader)).item()


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
    plt.close()

def test_fold(dataloader, class_names, model, loss_fn, file_name="resnet/v0.0", experiment_path="resnet", fold=0):

    # size of the dataset and number of batches
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # model in the testing process
    model.eval()

    test_loss, correct = 0, 0

    # initialize lists of predictions and labels for confusion matrix
    all_preds, all_labels, incorrect_examples, correct_examples = [], [], [], []

    # disable gradient computation
    with torch.no_grad():

        for i, (X, y) in enumerate(dataloader):
            # transforms the inputs to the device format (CPU or GPU)
            X, y = X.to(device), y.to(device)

            # prediction
            pred = model(X)

            # get the class with the highest probability as the prediction and append to the list
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            if preds != y:
                incorrect_examples.append(i)
            else:
                correct_examples.append(i)
            
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
        if not os.path.exists('folds/' + experiment_path + '/logs/'):
            os.makedirs('folds/' + experiment_path + '/logs/')
        with open('folds/' + experiment_path + '/logs/test_' + file_name + '.txt', 'w') as f:
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

    if not os.path.exists('folds/' + experiment_path + '/confusion_matrix/normalized/'):
        os.makedirs('folds/' + experiment_path + '/confusion_matrix/normalized/')
    plt.savefig('folds/' + experiment_path + '/confusion_matrix/normalized/test_normalized_' + file_name + '.png')
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Fold ' + str(fold+1))

    if not os.path.exists('folds/' + experiment_path + '/confusion_matrix/absolute/'):
        os.makedirs('folds/' + experiment_path + '/confusion_matrix/absolute/')
    plt.savefig('folds/' + experiment_path + '/confusion_matrix/absolute/test_absolute_' + file_name + '.png')
    plt.close()
    
    return all_preds, all_labels, incorrect_examples, correct_examples, accuracy

def get_losses_by_epoch(log_file_path):
    losses = []
    try:
        with open(log_file_path, 'r') as f:
            # if ".txt" in log_file_path:
            #     for line in f:
            #         if ':' in line:
            #             epoch, loss = line.strip().split(':')
            #             losses.append(float(loss))
            # elif ".json" in log_file_path:
            data = json.load(f)
            losses = data["losses"]
    except Exception as e:
        print(f"Error reading log file: {e}")
    return losses

def plot_training_loss(logs_path, n_folds=5, file_name="resnet", experiment_path="resnet/v0.0"):
    fig, folds = plt.subplots(math.ceil(n_folds/3.0), 3, figsize=(20, 12))
    i = 0
    all_losses_plot = folds.flat[n_folds]
    training_losses = []

    for fold in range(n_folds):
        log_file_path = os.path.join(logs_path, f'train_{file_name}_{fold+1}.json')
        training_loss = get_losses_by_epoch(log_file_path)
        training_losses.append(training_loss)

    for training_loss in training_losses:
        fold_plot = folds.flat[i]
        i = i+1

        # plot training loss curve
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
    if not os.path.exists('folds/' + experiment_path + '/results/'):
        os.makedirs('folds/' + experiment_path + '/results/')
    plt.savefig('folds/' + experiment_path + '/results/training_loss_' + file_name + '.png')
    plt.close()

def plot_complete_confusion_matrix(all_preds, all_labels, class_names, file_name="resnet", experiment_path="resnet/v0.0"):
    if not os.path.exists('folds/' + experiment_path + '/confusion_matrix/test/complete/'):
        os.makedirs('folds/' + experiment_path + '/confusion_matrix/test/complete/')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # save confusion matrix as image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Complete Confusion Matrix')

    plt.savefig('folds/' + experiment_path + '/confusion_matrix/test/complete/test_normalized_' + file_name + '.png')
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Complete Confusion Matrix')

    plt.savefig('folds/' + experiment_path + '/confusion_matrix/test/complete/test_absolute_' + file_name + '.png')
    plt.close()

def plot_accuracies(accuracies, file_name="resnet", experiment_path="resnet/v0.0"):
    folds_ticks = [i+1 for i in range(len(accuracies))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(1, len(accuracies) + 1), accuracies, marker='o', color='blue', s=100)
    ax.set_xticks(folds_ticks)
    ax.set_xlabel('Fold')
    ax.set_ylim(min(accuracies)-5, max(accuracies)+5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.grid(True)
    if not os.path.exists('folds/' + experiment_path + '/results/'):
        os.makedirs('folds/' + experiment_path + '/results/')
    plt.savefig('folds/' + experiment_path + '/results/test_accuracy_' + file_name + "_alt" + '.png')
    plt.close()

def get_img_dataset(img_index, datasets, fold):
    img_dataset = datasets[fold]
    if img_index >= len(datasets[fold]):
        img_index -= len(datasets[fold])
        img_dataset = datasets[-1]
    return img_dataset, img_index

def get_individual_by_path(path, individuals):
    for individual in individuals:
        if individual in path:
            return individual
    return None

def plot_incorrect_predictions_statistics(datasets, fold_preds, fold_incorrect_examples, class_names, individuals, file_name="resnet", experiment_path="resnet/v0.0", fold=0):
    nrows = 4
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        index = random.randint(0, len(fold_incorrect_examples)-1)
        img_index = fold_incorrect_examples[index]
        if i < nrows * ncols and i < len(fold_incorrect_examples):
            img_dataset, img_index = get_img_dataset(img_index, datasets, fold)
            path, label = img_dataset.samples[img_index]
            tensor = img_dataset[img_index][0].cpu()
            img = (tensor.numpy().transpose((1, 2, 0))).squeeze()
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1) # Unnormalize
            individual = get_individual_by_path(path, individuals)
            ax.imshow(img) # Use 'viridis' colormap for grayscale numpy arrays
            ax.set_title(f'Prev: {class_names[fold_preds[img_index]]}, Y: {class_names[label]}\nName: {individual}') # Set a title for each image
            ax.axis('off') # Hide the x and y axis ticks and labels for a cleaner look
            # images.append(img)
        else:
            # Hide any unused subplots if the number of images is less than nrows*ncols
            fig.delaxes(ax)
    
    plt.tight_layout(h_pad=2.0)
    
    os.makedirs('folds/' + experiment_path + '/incorrect_predictions/examples/', exist_ok=True)
    plt.savefig('folds/' + experiment_path + '/incorrect_predictions/examples/' + file_name + '.png')
    plt.close()

    images = []
    plt.title("Incorrect examples histogram")
    colors = ['red', 'green', 'blue']
    pixels = []
    for index in fold_incorrect_examples:
        img_dataset, img_index = get_img_dataset(index, datasets, fold)
        tensor = img_dataset[img_index][0].cpu()
        img = (tensor.numpy().transpose((1, 2, 0))).squeeze()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1) # Unnormalize
        if len(pixels) == 0:
            pixels = img
        else:
            pixels = pixels + img

    # for i, img in enumerate(images):
    #     if i == 0:
    #         continue
    #     pixels = pixels + img

    # Plot each channel
    for i, color in enumerate(colors):
        sns.histplot(pixels[:, :, i].ravel(), color=color, kde=True, label=color, element="step")

    plt.xlim(0, 255)
    # plt.ylim(0, 11000)
    plt.tight_layout(h_pad=2.0)
    plt.legend()
    plt.savefig('folds/' + experiment_path + '/incorrect_predictions/examples/histograms_' + file_name + '.png')
    plt.close()
        
    incorrect_counts_class = {class_name: 0 for class_name in class_names}
    incorrect_counts_individual = {individual: 0 for individual in individuals}
    incorrect_counts_color = {'P': 0, 'G': 0}

    if not os.path.exists('folds/' + experiment_path + '/logs/'):
        os.makedirs('folds/' + experiment_path + '/logs/')
    with open('folds/' + experiment_path + '/logs/test_' + file_name + '.txt', 'a') as f:
        f.write(f"\nIncorrect Predictions Paths Fold {str(fold+1)}:\n")
        for index in fold_incorrect_examples:
            img_dataset, index = get_img_dataset(index, datasets, fold)
            path, label = img_dataset.samples[index]
            f.write("\n" + path)
            incorrect_counts_class[class_names[label]] += 1
            individual = get_individual_by_path(path, individuals)
            if individual is not None:
                incorrect_counts_individual[individual] += 1
                if 'P' in individual:
                    incorrect_counts_color['P'] += 1
                elif 'G' in individual:
                    incorrect_counts_color['G'] += 1
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))

    # Plot incorrect predictions per class
    ax[0].bar(incorrect_counts_class.keys(), incorrect_counts_class.values())
    ax[0].set_xlabel('Classes')
    ax[0].set_ylabel('Number of Incorrect Predictions')
    ax[0].set_title('Incorrect Predictions per Class')
    plt.xticks(rotation=45)

    # Plot incorrect predictions per color
    ax[1].bar(incorrect_counts_color.keys(), incorrect_counts_color.values())
    ax[1].set_xlabel('Dye Color')
    ax[1].set_ylabel('Number of Incorrect Predictions')
    ax[1].set_title('Incorrect Predictions per Dye Color')
    plt.xticks(rotation=45)

    # Plot incorrect predictions per individual
    ax[2].bar(incorrect_counts_individual.keys(), incorrect_counts_individual.values())
    ax[2].set_xlabel('Individuals')
    ax[2].set_ylabel('Number of Incorrect Predictions')
    ax[2].set_title('Incorrect Predictions per Individual')
    plt.xticks(rotation=45)
    
    plt.suptitle('Incorrect Predictions Counts Fold ' + str(fold+1))
    plt.tight_layout()
    if not os.path.exists('folds/' + experiment_path + '/incorrect_predictions/statistics/'):
        os.makedirs('folds/' + experiment_path + '/incorrect_predictions/statistics/')
    plt.savefig('folds/' + experiment_path + '/incorrect_predictions/statistics/incorrect_counts_' + file_name + '.png')
    plt.close()

def plot_correct_predictions_statistics(datasets, fold_correct_examples, file_name="resnet", experiment_path="resnet/v0.0", fold=0):
    plt.title("Incorrect examples histogram")
    colors = ['red', 'green', 'blue']
    pixels = []
    unnormalize = Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    for index in fold_correct_examples:
        img_dataset, img_index = get_img_dataset(index, datasets, fold)
        tensor = unnormalize(img_dataset[img_index][0].cpu())
        img = (tensor.numpy().transpose((1, 2, 0))).squeeze()
        img = np.clip(img, 0, 1) # Unnormalize
        if len(pixels) == 0:
            pixels = img
        else:
            pixels = pixels + img

    # Plot each channel
    for i, color in enumerate(colors):
        sns.histplot(pixels[:, :, i].ravel(), color=color, kde=True, label=color, element="step")

    # plt.xlim(0, 255)
    # plt.ylim(0, 11000)
    plt.tight_layout(h_pad=2.0)
    plt.legend()
    os.makedirs('folds/' + experiment_path + '/correct_predictions/', exist_ok=True)
    plt.savefig('folds/' + experiment_path + '/correct_predictions/histograms_' + file_name + '.png')
    plt.close()

def load_model_weights(model, weight_path):
    state_dict = torch.load(weight_path, weights_only=True)
    model.load_state_dict(state_dict)
    # model.eval()
    return model

class EqualizeHistogram:
    def __init__(self):
        pass

    def __call__(self, x):
        return equalize(x)

def __main__():
    with open('config.json', 'r') as f:
        config = json.load(f)

    image_path = config["dataset"]["dataset_path"]

    class_names = config["dataset"]["class_names"]

    fold_individuals = config["dataset"]["folds"]
    for i in range(fold_individuals.__len__()):
        fold_individuals[i] = fold_individuals[i] + config["dataset"]["only_test"]
    num_folds = fold_individuals.__len__()
    fold_paths = [image_path + f"fold_{i+1}" for i in range(num_folds)]

    network_name = config["network"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    loss_fn = nn.CrossEntropyLoss()
    network_full_name = network_name + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_" + config["optimizer"]
    
    transform = Compose([
        # EqualizeHistogram(),
        Resize((224,224)),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    datasets_folds = []
    all_fold_preds = []
    all_fold_labels = []
    all_fold_accuracies = []

    # version = ""
    # weights_prefix = "folds/weights/" + network_full_name
    # if os.path.exists(weights_prefix + "_1.pth"):
    #     for i in range(1, 100):
    #         if not os.path.exists(weights_prefix + "_1_v" + str(i) + ".pth"):
    #             version = "_v" + str(i)
    #             break

    version = config["experiment_version"]
    if version == None:
        version = ""
    
    experiment_path = network_full_name + "/" + version
    starting_fold = parser.parse_args().starting_fold
    ending_fold = parser.parse_args().ending_fold

    if os.path.exists("folds/weights/" + experiment_path) and not config["quick_test"]["enabled"]:
        confirm_version = None
        for fold in range(starting_fold, ending_fold+1):
            for weight in os.listdir("folds/weights/" + experiment_path):
                if str(fold) + ".pth" in weight:
                    confirm_version = input(f"Version {version} of fold {fold} already exists. Do you want to overwrite it? (y/n) ")
                    break
            if confirm_version != None and confirm_version.lower() != 'y':
                print("Exiting program. Please change the version and try again.")
                exit(0)
    else:
        os.makedirs("folds/weights/" + experiment_path, exist_ok=True)
    

    for fold in range(num_folds):
        datasets_folds.append(datasets.ImageFolder(root=fold_paths[fold], transform=transform, target_transform=None))
    datasets_folds.append(datasets.ImageFolder(root=image_path + "only_test", transform=transform, target_transform=None))

    for fold in range(starting_fold-1, ending_fold):
        print(f"================ Fold {fold+1} / {ending_fold} ================")
        train_dataset = ConcatDataset([datasets_folds[i] for i in range(num_folds) if i != fold])
        test_dataset = ConcatDataset([datasets_folds[fold], datasets_folds[-1]])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        file_name = network_name + "_" + str(fold+1)

        #! usado apenas em testes rápidos sem treinamento
        if config["quick_test"]["enabled"]:
            print("Quick test, ignoring specified neural network and using resnet34.")
            test_version = config["quick_test"]["version"]
            if config["quick_test"]["weights_version"] == "":
                experiment_path = config["quick_test"]["folder_name"]
            else:
                experiment_path = config["quick_test"]["folder_name"] + "/" + config["quick_test"]["weights_version"]
            test_path = "/quick_test/" + experiment_path
            weight_path = config["quick_test"]["weights_path"] + experiment_path

            file_name = "resnet_" + str(fold+1)
            resnet = models.resnet34().to(device)
            resnet = load_model_weights(resnet, weight_path + "/" + file_name + ".pth")
            fold_preds, fold_labels, fold_incorrect_examples, fold_correct_examples, fold_accuracy = test_fold(test_dataloader, class_names, resnet, loss_fn, file_name=file_name, experiment_path=test_path, fold=fold)
            if config["generate_statistics"]:
                plot_correct_predictions_statistics(datasets_folds, fold_correct_examples, file_name=file_name, experiment_path=test_path, fold=fold)
                plot_incorrect_predictions_statistics(datasets_folds, fold_preds, fold_incorrect_examples, class_names, fold_individuals[fold], file_name=file_name, experiment_path=test_path, fold=fold)

            all_fold_preds.extend(fold_preds)
            all_fold_labels.extend(fold_labels)
            all_fold_accuracies.append(fold_accuracy)
            continue

        model = None
        if network_name == "resnet34" or network_name == "resnet":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
        elif network_name == "vgg16" or network_name == "vgg":
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
        elif network_name == "densenet121" or network_name == "densenet":
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)
        else:
            print("Neural network architecture not recognized. Try again.")
            exit(1)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        start, end = 0, 0
        training_loss = []

        for t in range(epochs):
            print(f"-------------------------------\nEpoch {t+1}")
            start = time.time()
            loss = train_fold(train_dataloader, model, loss_fn, (t+1), optimizer, file_name=file_name, experiment_path=experiment_path)
            end = time.time()
            print(f"Epoch time: {end-start}")
            training_loss.append(loss)

        # training_losses.append(training_loss)
        if not os.path.exists("folds/weights/" + experiment_path):
            os.makedirs("folds/weights/" + experiment_path)
        torch.save(model.state_dict(), "folds/weights/" + experiment_path + "/" + file_name + ".pth")

        start = time.time()
        fold_preds, fold_labels, fold_incorrect_examples, fold_correct_examples, fold_accuracy = test_fold(test_dataloader, class_names, model, loss_fn, file_name=file_name, experiment_path=experiment_path, fold=fold)
        end = time.time()
        print(f"Test time: {end-start}")

        if config["generate_statistics"]:
            plot_incorrect_predictions_statistics(datasets_folds, fold_preds, fold_incorrect_examples, class_names, fold_individuals[fold], file_name=file_name, experiment_path=experiment_path, fold=fold)

        if not os.path.exists('folds/' + experiment_path + '/logs/'):
            os.makedirs('folds/' + experiment_path + '/logs/')

        with open('folds/' + experiment_path + '/logs/train_' + file_name + '.json', 'w') as f:
            json.dump({
                "losses": training_loss
            }, f, indent=4)

        with open('folds/' + experiment_path + '/logs/test_' + file_name + '.json', 'w') as f:
            json.dump({
                "accuracy": fold_accuracy,
                "predictions": fold_preds.tolist(),
                "labels": fold_labels.tolist()
            }, f, indent=4)
        # all_fold_preds.extend(fold_preds)
        # all_fold_labels.extend(fold_labels)
        # all_fold_accuracies.append(fold_accuracy)

    if not config["quick_test"]["enabled"] and ending_fold == num_folds:
        plot_training_loss('folds/' + experiment_path +'/logs/', file_name=network_name, experiment_path=experiment_path)
        
        all_fold_accuracies = []
        all_fold_preds = []
        all_fold_labels = []
        
        for log in os.listdir('folds/' + experiment_path +'/logs/'):
            print(f"Processing log file: {log}")
            if "test" in log and log.endswith(".json"):
                with open('folds/' + experiment_path +'/logs/' + log, 'r') as f:
                    data = json.load(f)
                    all_fold_accuracies.append(data["accuracy"])
                    all_fold_preds.extend(np.array(data["predictions"]))
                    all_fold_labels.extend(np.array(data["labels"]))
        plot_accuracies(all_fold_accuracies, file_name=file_name, experiment_path=experiment_path)

        plot_complete_confusion_matrix(all_fold_preds, all_fold_labels, class_names, file_name=file_name, experiment_path=experiment_path)

    elif config["quick_test"]["enabled"]:
        plot_complete_confusion_matrix(all_fold_preds, all_fold_labels, class_names, file_name=file_name, experiment_path=experiment_path)
if __name__ == "__main__":
    __main__()