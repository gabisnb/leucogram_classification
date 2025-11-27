import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
parser.add_argument("-c", "--check-dataset", action="store_true", help="Check dataset structure and classes.")

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} files in "{dirpath}".')


def check_dataset(train_data, test_data, val_data):
    print(f'Train data: \n{train_data} \nTest data: \n{test_data} \nValidation data: \n{val_data}')

    class_names = train_data.classes
    print(f'Class names: {class_names}')

    class_dict = train_data.class_to_idx
    print(f'Class to index mapping: {class_dict}')

    print(f'Number of samples per set: {len(train_data), len(test_data), len(val_data)}')


def train(dataloader, model, loss_fn, epoch, lr, optimizer, version = "", training_loss=None):

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
        with open('logs/train' + str(batch_size) + '_' + str(lr) + '_' + str(parser.parse_args().epochs) + '_sgd' + version +'.txt', 'a') as f:
            f.write(f"Training Error Epoch {epoch}: {totalLoss/len(dataloader):>7f} \n\n")
    except:
        print("Error writing to file")


def validate(dataloader, class_names, model, loss_fn, batch_size, epoch, lr, optimizer, version = "", validation_loss=None, validation_accuracy=None):
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

def test(dataloader, class_names, model, loss_fn, batch_size, epoch, lr, optimizer, version = ""):

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
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # save results to file
    try:
        with open('logs/test_' + str(batch_size) + '_' + str(epoch) + '_' + str(lr) + '_' + str(optimizer) + version + '.txt', 'w') as f:
            f.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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

    plt.savefig('confusion_matrix/test/test_cm_' + str(batch_size) + '_' + str(epoch) + '_' + str(lr) + '_' + str(optimizer) + version + '_test.png')


data_path = Path("/home/gabriela/projetos/datasets/")
image_path = data_path / "cut_bboxes_separated"

if image_path.is_dir():
    print(f'The directory "{image_path}" exists.')
else:
    print(f'The directory "{image_path}" does not exist.')
    exit()

if parser.parse_args().check_dataset:
    walk_through_dir(image_path)

train_dir = image_path / "train"
val_dir = image_path / "val"
test_dir = image_path / "test"

# random.seed(198)
# image_path_list = list(image_path.glob('*/*/*.jpg'))
# random_image_path = random.choice(image_path_list)
# image_class = random_image_path.parent.stem

# img = Image.open(random_image_path)
# print(f'Random image path: {random_image_path}')
# print(f'Image class: {image_class}')
# print(f'Image height: {img.height}')
# print(f'Image width: {img.width}')
# img.save("random_image.jpg")
      
transform = Compose([
    Resize((224,224)),
    ToTensor(),
    Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=train_dir, transform=transform, target_transform=None)

test_data = datasets.ImageFolder(root=test_dir, transform=transform, target_transform=None)

val_data = datasets.ImageFolder(root=val_dir, transform=transform, target_transform=None)

if parser.parse_args().check_dataset:
    check_dataset(train_data, test_data, val_data)

num_classes = 5
epochs = parser.parse_args().epochs
batch_size = 16
learning_rate = 0.01
loss_fn = nn.CrossEntropyLoss()
class_names = train_data.classes

train_dataloader = DataLoader(dataset = train_data, batch_size=batch_size, num_workers=1, shuffle=True)
test_dataloader = DataLoader(dataset = test_data, batch_size=batch_size, num_workers=1, shuffle=False)
val_dataloader = DataLoader(dataset = val_data, batch_size=batch_size, num_workers=1, shuffle=False)

# resnet = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
resnet = models.resnet34(pretrained=True).to(device)
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

version = ""
if os.path.exists("weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd.pth"):
    for i in range(1, 100):
        if not os.path.exists("weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd_v" + str(i) + ".pth"):
            version = "_v" + str(i)
            break

start, end = 0, 0
training_loss = []
validation_loss = []
validation_accuracy = []

for t in range(epochs):
    print(f"-------------------------------\nEpoch {t+1}")
    start = time.time()
    train(train_dataloader, resnet, loss_fn, (t+1), learning_rate, optimizer, version=version, training_loss=training_loss)
    end = time.time()
    print(f"Epoch time: {end-start}")
    print("Running Validation...")
    start = time.time()
    validate(val_dataloader, class_names, resnet, loss_fn, batch_size, (t+1), learning_rate, 'sgd', version=version, validation_loss=validation_loss, validation_accuracy=validation_accuracy)
    end = time.time()
    print(f"Validation time: {end-start}")

if os.path.exists("weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd.pth"):
    for i in range(1, 100):
        if not os.path.exists("weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd_v" + str(i) + ".pth"):
            torch.save(resnet.state_dict(), "weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd_v" + str(i) + ".pth")
            break
else:
    torch.save(resnet.state_dict(), "weights/resnet_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + "_sgd.pth")

training_loss = [loss.cpu().item() for loss in training_loss]

# plot training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PyTorch Training Loss Curve')
plt.grid(True)
plt.savefig('results/training_loss_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epochs) + version + '_sgd.png')

# plot validation loss and accuracy curves
fig, ax = plt.subplots(1, 2)
loss_plot, accuracy_plot = ax

loss_plot.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss (PyTorch)', color='orange')
loss_plot.set_xlabel('Epoch')
loss_plot.set_ylabel('Loss')
loss_plot.set_title('PyTorch Validation Loss Curve')
loss_plot.grid(True)

accuracy_plot.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy (PyTorch)', color='green')
accuracy_plot.set_xlabel('Epoch')
accuracy_plot.set_ylabel('Accuracy (%)')
accuracy_plot.set_title('PyTorch Validation Accuracy Curve')
accuracy_plot.grid(True)
plt.tight_layout()
plt.savefig('results/validation_loss_accuracy_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epochs) + version + '_sgd.png')

#### Testing
start = time.time()
test(test_dataloader, class_names, resnet, loss_fn, batch_size, (t+1), learning_rate, 'sgd', version=version)
end = time.time()
print(f"Test time: {end-start}")
print("\nDone!")