from scipy.io import loadmat
import numpy as np

def load_train():
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension
    targets = data['tr_labels']
    identities = data['tr_identity']
    # train_inputs = np.vstack(images).reshape(images.shape[0], images.shape[1]*images.shape[2])
    train_inputs = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    train_targets = np.squeeze(data['tr_labels'])
    return train_inputs, train_targets
