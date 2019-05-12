import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def save_model_results(model_name, description, num_epochs, num_classes, train_size, val_size, train_accuracies, val_acccuracies, train_losses, val_losses):
    with open('model_results/all_results.csv', 'a+') as f:
        write_str = str(model_name) +',' +str(description) +',' +str(num_epochs) +',' +str(num_classes) +',' +str(train_size) +',' +str(val_size) +',' +str(train_accuracies[-1]) +',' +str(val_accuracies[-1]) +',' +str(train_losses[-1]) +',' +str(val_losses[-1]) +','+str(mean(train_accuracies)) +',' +str(mean(val_accuracies)) +',' +str(mean(train_losses)) +',' +str(mean(val_losses[-1])) 
        f.write(write_str+'\n')
        
def plot_model_stats(train_accuracies, val_accuracies, train_losses, val_losses, save_path, latent_scores=None):
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(val_losses, 'bo-', label = 'val-loss')
    plt.plot(train_losses, 'ro-', label = 'train-loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'bo-', label = 'val-acc')
    plt.plot(train_accuracies, 'ro-', label = 'train-acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='lower right')
    print('Saving figure to ' + save_path)
    plt.savefig(save_path)
    plt.show()
    
    
    
def plot_confusion_matrix(graph_array, labels, path):
    graph_array = [[int(i) for i in j ] for j in graph_array]
    columns = [str(i[0])+": "+str(i[1]*100) for i in labels]
    rows=[str(i[0]) for i in labels]
    print(len(columns))
    print(len(graph_array))
    print(graph_array)

    df_cm = pd.DataFrame(graph_array, columns =columns , index = rows)
    
    plt.figure(figsize = (10,7))
    sns.set(font_scale=2)

    sns.heatmap(df_cm, annot=True, fmt="d") 
    plt.savefig('model_results/confusion_'+path+'.png')

    #df_cm = pd.DataFrame(validation_array, index = [i for i in labels],
    #              columns = [i for i in labels])
    #plt.figure(figsize = (10,7))
    #sns.heatmap(df_cm, annot=True, fmt="d")