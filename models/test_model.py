from data_processing.datasets import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from plot_model_stats import plot_confusion_matrix
from train_model import my_collate

#Use this function when the train dataset and test dataset come from different folders

def test_model(model, testset, trainset, batch_size=3, device='cuda:0', joint_run = False):
    # Assumes testset and trainset have the same labels
    def write_results(conf, conf_acc, acc):
        with open('model_results/'+model.title+'.txt', 'a+') as f:
            f.write(str(conf)+'\n')
            f.write(str(conf_acc)+'\n')
            f.write(str(acc)+'\n')
        
    assert trainset.labels == testset.labels
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=batch_size,
                                             collate_fn=joint_collate if joint_run else my_collate)

    model.to(device)
    correct = 0.0
    cum_loss = 0.0
    num_trained= 1.0
    confusion_matrix=torch.zeros(len(trainset.labels), len(trainset.labels))
    print(trainset.labels)
    print("true", "predicted")
    for (i, (inputs, labels)) in enumerate(test_loader):
        try:
            if joint_run:
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            scores = model(inputs)
            scores = F.softmax(scores)
            max_scores, max_labels = scores.max(1)
            correct_eval = (max_labels == labels).sum().item()
            correct+= correct_eval
            num_trained+=batch_size
            _, preds = torch.max(scores, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            if (i+1) %10 == 0:
                conf_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
                acc = correct/num_trained
                write_results(confusion_matrix, conf_acc, acc)
                print(confusion_matrix)
                print(conf_acc)
                print(acc)
        except Exception as e:
            print(e)
    conf_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    acc = correct/num_trained
    write_results(confusion_matrix, conf_acc, acc)
    print('Confusion...')
    plot_confusion_matrix(confusion_matrix.cpu().numpy(), zip(trainset.labels,conf_acc.cpu().numpy().tolist()), model.title)
    print(confusion_matrix)
    print(conf_acc)
    print(acc)
