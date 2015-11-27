from scipy.io import loadmat
import numpy as np

def load_data():
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']
    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    # targets = np.squeeze(data['tr_labels'])

    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    num_unlabeled = sum(targets_sort[:,1] == -1)
    valid_targets = targets_sort[0:num_unlabeled,0]
    train_targets = targets_sort[(num_unlabeled + 1):,0]

    # Sort the images
    temp = np.append(images,identities,1)
    images_sort = temp[temp[:,-1].argsort()]
    valid_inputs = images_sort[0:num_unlabeled,0:-1] # Tested that the -1 omits the identity last column
    train_inputs = images_sort[(num_unlabeled + 1):,0:-1]

    return train_inputs, train_targets, valid_inputs, valid_targets
