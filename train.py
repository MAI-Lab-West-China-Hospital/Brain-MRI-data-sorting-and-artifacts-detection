# https://github.com/Project-MONAI/tutorials/blob/master/3d_classification/torch/densenet_training_dict.py
import logging
import os
import sys
import time
import csv
import argparse
import ssl

from sklearn.model_selection import train_test_split
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import monai
from monai.networks.nets import DenseNet121, resnet50, EfficientNetBN, SEResNet50, SENet154
from EfficientNet import MyEfficientNet
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import *
from monai.utils import set_determinism
from monai.data import CSVSaver

from ImageFolder import make_dataset
from dataset import MRIDataset
ssl._create_default_https_context=ssl._create_unverified_context
set_determinism(seed=2021)
warnings.filterwarnings('ignore')


def form_results(lr, epochs, stack, model):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = './Results/' + model
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_stack{1}_lr{2}_epochs{3}_pretrainTrue".format(date, stack, lr, epochs)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def train(train_loader, val_loader, pretrained, architecture: str, classes, stack, epochs):
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # chose model
    # assert (architecture in ['resnet50', 'EfficientNetBN', 'EfficientNet', 'densenet121', 'SENet154', 'SEResNet50'])
    if architecture == 'resnet50':
        model = resnet50(spatial_dims=2, n_input_channels=stack, num_classes=len(classes),
                         pretrained=pretrained).to(device)

    elif architecture == 'densenet121':
        model = DenseNet121(spatial_dims=2, in_channels=stack, out_channels=len(classes),
                            pretrained=pretrained).to(device)

    elif architecture == 'EfficientNetBN':
        model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=stack,
                               num_classes=len(classes), pretrained=pretrained).to(device)

    elif architecture == 'EfficientNet':
        model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=stack,
                               num_classes=len(classes), pretrained=pretrained, dropout_rate=0.2).to(device)

    elif architecture == 'SEResNet50':
        model = SEResNet50(spatial_dims=2, in_channels=stack, num_classes=len(classes),
                           pretrained=pretrained).to(device)

    elif architecture == 'SENet154':
        model = SENet154(spatial_dims=2, in_channels=stack, num_classes=len(classes),
                         pretrained=pretrained).to(device)

    elif architecture == 'vgg16':
        model = torchvision.models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(classes))
        model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    auc_metric = ROCAUCMetric()

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    learning_rate = []  # to save lr of each epoch
    train_writer = SummaryWriter(os.path.join(tensorboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(tensorboard_path, 'val'))

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if (step + 1) % 100 == 0:
                print(f"{step + 1}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        train_writer.add_scalar("Epoch Loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # scheduler.step()  # if cosine
        # learning_rate.append(scheduler.get_last_lr()[0])

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                val_loss = loss_function(y_pred, y)
                val_writer.add_scalar("Epoch Loss", val_loss.item(), epoch + 1)
                # calculate accuracy
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                y_onehot = [post_label(i) for i in decollate_batch(y)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]  # one-hot format

                auc_metric(y_pred_act, y_onehot)  # use one-hot format for computing auc
                auc_result = auc_metric.aggregate()

                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), saved_model_path + "/best_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                val_writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

            scheduler.step(val_loss)
            learning_rate.append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
            }, saved_model_path + f'/model_{epoch + 1}.pth')
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    train_writer.close()
    val_writer.close()

    # save learning rate
    with open(os.path.join(log_path, 'lr_change.txt'), 'w') as f:
        f.write(str(learning_rate))


def test(architecture, test_loader, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # chose model
    # assert (architecture in ['resnet50', 'EfficientNetBN', 'EfficientNet', 'densenet121', 'SENet154', 'SEResNet50'])
    if architecture == 'resnet50':
        model = resnet50(spatial_dims=2, n_input_channels=1, num_classes=len(classes)).to(device)
    if architecture == 'densenet121':
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(classes)).to(device)
    if architecture == 'EfficientNetBN':
        model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=1, num_classes=len(classes)).to(device)
    if architecture == 'SEResNet50':
        model = SEResNet50(spatial_dims=2, in_channels=1, num_classes=len(classes)).to(device)
    if architecture == 'SENet154':
        model = SENet154(spatial_dims=2, in_channels=1, num_classes=len(classes)).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir=output_path)
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
            test_outputs = model(test_images).argmax(dim=1)
            value = torch.eq(test_outputs, test_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(test_outputs, test_data["name"])
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        saver.finalize()
    pass


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='training parameters')

    parser.add_argument('--cuda', default='0', type=str, help='gpu id')
    parser.add_argument('--dataroot', default='../MRIseq_classification/data')

    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='entire epoch number')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--stack', default=3, type=int, help='input channel')

    parser.add_argument('--model', default='EfficientNet', help='choose model')
    # architecture can be ['resnet50', 'EfficientNetBN', 'densenet121', 'SENet154', 'SEResNet50', 'vgg16']
    parser.add_argument('--pretrained', default=True, type=bool)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.gz')

    # get address
    tensorboard_path, saved_model_path, log_path = form_results(lr=args.lr,
                                                                epochs=args.epoch,
                                                                stack=args.stack,
                                                                model=args.model)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # get data
    img_label_dict, imgs, labels, classes, class_to_idx = make_dataset(args.dataroot, extensions=extensions)
    print(f'There are {len(classes)} classes, specifically：')
    print(class_to_idx)

    # save classes_to_idx into files
    csv_path = os.path.join(log_path, 'classes.csv')
    with open(csv_path, 'a') as file:
        w = csv.writer(file)
        for key, val in class_to_idx.items():
            # write every key and value to file
            w.writerow([key, val])

    # save args
    argsDict = args.__dict__
    argsPath = os.path.join(log_path, 'args.txt')
    with open(argsPath, 'w') as f:
        f.writelines('------------------ start --------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------------ end --------------' + '\n')

    # split train val and test = 6:2:2
    X_train_val, X_test, y_train_val, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42,
                                                          stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42,
                                                                stratify=y_train_val)
    # stratify = labels, make sure every class into test dataset

    # count the number of each class
    trainset = set(y_train)
    train_count = {}
    for cls in trainset:
        train_count.update({cls: [y_train.count(cls), labels.count(cls)]})
    print('number of train data/all data')
    print(train_count)

    valset = set(y_val)
    val_count = {}
    for cls in valset:
        val_count.update({cls: [y_val.count(cls), labels.count(cls)]})
    print('number of val data/all data')
    print(val_count)

    # build train files and val files
    train_data = []
    val_data = []
    for i, j in zip(X_train, y_train):
        train_data.append({'img': i, 'label': j})

    for i, j in zip(X_val, y_val):
        val_data.append({'img': i, 'label': j})

    # Define transforms for image
    train_transforms = Compose(
        [
            ScaleIntensityd(keys='img'),
            RandGaussianNoised(keys='img', prob=0.5, mean=0.0, std=0.1),
            RandFlipd(keys='img', prob=0.2, spatial_axis=0),
            RandFlipd(keys='img', prob=0.2, spatial_axis=1),
            Rand2DElasticd(keys='img', prob=0.2, spacing=(20, 20), magnitude_range=(1, 2)),
            EnsureTyped(keys=['img', 'label'])
        ]
    )
    val_transforms = Compose(
        [
            ScaleIntensityd(keys='img'),
            EnsureTyped(keys=['img', 'label'])
        ]
    )

    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=len(classes))])

    # # Define dataset, data loader
    check_ds = MRIDataset(data=train_data[:10], stack=args.stack, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print('check image shape and label: ')
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = MRIDataset(data=train_data, stack=args.stack, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                              pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = MRIDataset(data=val_data, stack=args.stack, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=4, pin_memory=torch.cuda.is_available())

    train(train_loader, val_loader, args.pretrained, args.model, classes, args.stack, args.epoch)