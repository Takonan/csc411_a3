from scipy.io import loadmat
import numpy as np

def load_data(include_mirror=False):
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']

    # Generate a mirrored version if necessary:
    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
        identities = np.append(identities, identities,0)
        targets = np.append(targets, targets,0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

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

def load_data_with_test(include_mirror=False):
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']

    # Generate a mirrored version if necessary:
    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
        identities = np.append(identities, identities,0)
        targets = np.append(targets, targets,0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    # targets = np.squeeze(data['tr_labels'])

    # # Sort the array based on the tr_identities
    # # Sort the targets
    # temp = np.append(targets,identities,1)
    # targets_sort = temp[temp[:,1].argsort()]
    # num_unlabeled = sum(targets_sort[:,1] == -1)
    # valid_targets = targets_sort[0:num_unlabeled,0]
    # test_targets = targets_sort[(num_unlabeled + 1):2*num_unlabeled,0]
    # train_targets = targets_sort[(2*num_unlabeled + 1):,0]

    # # Sort the images
    # temp = np.append(images,identities,1)
    # images_sort = temp[temp[:,-1].argsort()]
    # valid_inputs = images_sort[0:num_unlabeled,0:-1] # Tested that the -1 omits the identity last column
    # test_inputs = images_sort[(num_unlabeled + 1):2*num_unlabeled,0:-1]
    # train_inputs = images_sort[(2*num_unlabeled + 1):,0:-1]

    # DEBUG: Only use half of the unlabeled identity for validation
    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    num_unlabeled = sum(targets_sort[:,1] == -1)
    valid_targets = targets_sort[0:num_unlabeled/2,0]
    test_targets = targets_sort[(num_unlabeled/2 + 1):num_unlabeled,0]
    train_targets = targets_sort[(num_unlabeled + 1):,0]

    # Sort the images
    temp = np.append(images,identities,1)
    images_sort = temp[temp[:,-1].argsort()]
    valid_inputs = images_sort[0:num_unlabeled/2,0:-1] # Tested that the -1 omits the identity last column
    test_inputs = images_sort[(num_unlabeled/2 + 1):num_unlabeled,0:-1]
    train_inputs = images_sort[(num_unlabeled + 1):,0:-1]
    # End debug

    return train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets

def load_data_one_of_k(include_mirror=False):
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']

    # Generate a mirrored version if necessary:
    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
        identities = np.append(identities, identities,0)
        targets = np.append(targets, targets,0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    # targets = np.squeeze(data['tr_labels'])

    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    num_unlabeled = sum(targets_sort[:,1] == -1)

    temp = np.zeros((temp.shape[0], 7))
    for i in range(temp.shape[0]):
        temp[i][targets_sort[i,0]-1] = 1

    valid_targets = temp[0:num_unlabeled,:]
    train_targets = temp[(num_unlabeled+1):,:]

    # Sort the images
    temp = np.append(images,identities,1)
    images_sort = temp[temp[:,-1].argsort()]
    valid_inputs = images_sort[0:num_unlabeled,0:-1] # Tested that the -1 omits the identity last column
    train_inputs = images_sort[(num_unlabeled + 1):,0:-1]

    return train_inputs, train_targets, valid_inputs, valid_targets


def load_data_with_test_one_of_k(include_mirror=False):
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']

    # Generate a mirrored version if necessary:
    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
        identities = np.append(identities, identities,0)
        targets = np.append(targets, targets,0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    # targets = np.squeeze(data['tr_labels'])

    # DEBUG: Only use half of the unlabeled identity for validation
    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    num_unlabeled = sum(targets_sort[:,1] == -1)

    temp = np.zeros((temp.shape[0], 7))
    for i in range(temp.shape[0]):
        temp[i][targets_sort[i,0]-1] = 1

    valid_targets = temp[0:num_unlabeled/2,:]
    test_targets = temp[(num_unlabeled/2+1):num_unlabeled,:]
    train_targets = temp[(num_unlabeled+1):,:]

    # Sort the images
    temp = np.append(images,identities,1)
    images_sort = temp[temp[:,-1].argsort()]
    valid_inputs = images_sort[0:num_unlabeled/2,0:-1] # Tested that the -1 omits the identity last column
    test_inputs = images_sort[(num_unlabeled/2 + 1):num_unlabeled,0:-1]
    train_inputs = images_sort[(num_unlabeled + 1):,0:-1]
    # End debug

    return train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets
