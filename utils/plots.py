import torch
import matplotlib.pyplot as plt
import numpy as np

def inspect_data(loader):
    preview_dim = 20
    fig, axs = plt.subplots(preview_dim, preview_dim, figsize = (20, 20))
    
    for i in range(preview_dim):
        for j in range(preview_dim):
            features, labels = next(iter(loader))
            idx = torch.randint(0, features.shape[0] - 1, (1,))
            axs[i, j].axis("off")
            axs[i, j].imshow(features[idx].reshape(28, 28), cmap='gray')

    plt.show()

def draw_loss_graphs(train_loss_seq, test_loss_seq):
	plt.plot(train_loss_seq, label = 'Training loss')
	plt.plot(test_loss_seq, label = 'Test loss')
	plt.legend()

def inspect_misclassified(net, features_misclass, labels_misclass):
    n = min(len(features_misclass), 200)
    idx = 0
    ncol = 12
    nrow = int(n / 12) + 1
    
    fig, axs = plt.subplots(nrow, ncol, figsize = (20, 30))
    
    with torch.no_grad():
        net.eval()
        for i in range(nrow):
            for j in range(ncol):  
                if idx < n:
                    img = features_misclass[idx]
                    label = labels_misclass[idx]
                    output = net(img.reshape(1, 1, 28, 28))
                    _, preds = output.max(1)
                    
                    axs[i, j].imshow(img.reshape(28, 28), cmap = 'gray')
                    axs[i, j].set_title(str(preds.item()) + "\n actual: " + str(label.item()))
                axs[i, j].axis("off")
                idx += 1

        plt.show()