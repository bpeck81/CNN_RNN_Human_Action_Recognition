import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from plot_model_stats import *
import os
import matplotlib.pyplot as plt
import time
import torchnet
from torchnet.meter import ConfusionMeter
import torch.nn.functional as F


def my_collate(batch):
    # Filters DataLoader items after they are retrieved
    data = [item[0] for item in batch if item != None]
    target = [item[1] for item in batch if item != None]
    if len(data) == 0: return [None, None]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]


def train_model(model, loss_fn, batch_size, dataset, optimizer, model_title='None', device='cpu', root_dir='', num_epochs = 2, testset=None):
    def write_results(title, write_string):
        with open('model_results/'+title + '_results.txt', 'a+') as f:
            print(write_string)
            f.write(write_string + '\n')

    # Shuffling is needed in case dataset is not shuffled by default.
    train_ratio=.9
    train_len = int(len(dataset)*train_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len]) 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=my_collate
                                               )
    # We don't need to bach the validation set but let's do it anyway.
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             collate_fn=my_collate
                                             )  

    # GPU enabling.
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Training loop. Please make sure you understand every single line of code below.
    # Go back to some of the previous steps in this lab if necessary.
    start_time = time.time()
    plot_path=os.path.join(root_dir,'model_results/'+model_title+'.png')
    print("Trainset Length: ", len(train_loader))
    print("Valset Length: ", len(val_loader))
    print("Epoch Count: ", num_epochs)
    print("Plot Path: ",plot_path)
    
    train_accuracies=[]; val_accuracies=[]; train_losses=[]; val_losses=[]
    for epoch in range(0, num_epochs):
        correct = 0.0
        cum_loss = 0.0

        # Make a pass over the training data.
        model.train()
        num_trained = 1.0
        confusion_matrix = torch.zeros(len(dataset.labels), len(dataset.labels))
        for (i, (inputs, labels)) in enumerate(train_loader):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)#[0]

                # Forward pass. (Prediction stage)
                scores = model(inputs)
                loss = loss_fn(scores, labels)

                # Zero the gradients in the network.
                optimizer.zero_grad()

                # Backward pass. (Gradient computation stage)
                loss.backward()

                # Parameter updates (SGD step) -- if done with torch.optim!
                optimizer.step()

                scores = F.softmax(scores)
                max_scores, max_labels = scores.max(1)
                correct_eval = (max_labels == labels).sum().item()
                correct+= correct_eval
                num_trained+=batch_size
                cum_loss += loss.item()
                _, preds = torch.max(scores, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1


                # Logging the current results on training.
                if (i + 1) % 100 == 0:
                    print(dataset.labels)
                    write_string =('Train-epoch %d. Iteration %05d, Avg-Loss: %.4f, Accuracy: %.4f' %
                          (epoch, num_trained + 1, cum_loss / (num_trained), correct / (num_trained)))
                    write_string += '\n'+str(confusion_matrix)
                    confusion_avg = (confusion_matrix.diag()/confusion_matrix.sum(1))
                    write_string += '\n'+ str(confusion_avg)
                    write_results(model_title, write_string)
                    #print("Time Elapsed Minutes: ", (time.time() - start_time) /60)
                    correct = 0.0
                    cum_loss = 0.0
                    num_trained = 1.0
                    confusion_matrix = torch.zeros(len(dataset.labels), len(dataset.labels))
                    model_save_path= 'saved_models/'+model_title+'.pth'

                    torch.save(model.state_dict(), model_save_path)
            except Exception as e:
                
                print(e)
        train_accuracies.append(correct / num_trained)
        train_losses.append(cum_loss / num_trained)
        write_string = "Time Elapsed Minutes: "+ str((time.time() - start_time) /60)
        write_results(model_title, write_string)

        # Make a pass over the validation data.
        model.eval()
        num_trained = 1.0
        correct = 0.0
        cum_loss = 0.0
        confusion_matrix = torch.zeros(len(dataset.labels), len(dataset.labels))
        print('Validating...')
        print(dataset.labels)
        for (i, (inputs, labels)) in enumerate(val_loader):
            try:
  
                inputs = inputs.to(device)
                labels = labels.to(device)#[0]

                # Forward pass. (Prediction stage)
                scores = model(inputs)
                latent_scores.append([scores.item(),labels.item()])
                loss = loss_fn(scores, labels)
                scores = F.softmax(scores)
                # Count how many correct in this batch.
                max_scores, max_labels = scores.max(1)
                correct_eval = (max_labels == labels).sum().item()
                correct+= correct_eval
                num_trained+=batch_size
                cum_loss += loss.item()
                _, preds = torch.max(scores, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                if i % 100 == 0:
                    print(confusion_matrix)
            except Exception as e:
                print(e)
        val_accuracies.append(correct / num_trained)
        val_losses.append(cum_loss / num_trained)
        write_string =('validation-epoch %d. Iteration %05d, Avg-Loss: %.4f, Accuracy: %.4f' %
              (epoch, num_trained + 1, cum_loss / num_trained, correct / num_trained ))
        write_string += '\n'+str(confusion_matrix)
        confusion_avg = (confusion_matrix.diag()/confusion_matrix.sum(1))
        write_string += '\n'+ str(confusion_avg)
        write_results(model_title, write_string)
        plot_model_stats(train_accuracies, val_accuracies, train_losses, val_losses, plot_path, latent_scores=latent_scores)
        if testset != None:
            try:
                from test_model import test_model
                test_model(model,testset,dataset, batch_size, joint_run=joint_run)
            except Exception as e:
                print(e)

