from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def save_output_csv(filename, pred):
    """Save the prediction in a filename."""
    with open(filename, 'w') as f_result:
        f_result.write('Id,Prediction\n')
        for i, y in enumerate(pred, 1):
            f_result.write('{},{}\n'.format(i,y))
        num_entries = 1253
        for i in range(len(pred)+1, num_entries+1):
            f_result.write('{},{}\n'.format(i,0))
    return

def region_std(source,rec_size):
    # Compute the standard deviation in the 2*rec_size + 1 x 2*rec_size+1 square around each pixel
    row, col = source.shape
    output = np.zeros(source.shape)
    for r in range(row):
        for c in range(col):
            output[r,c] = np.std(source[max(0,r-rec_size):min(row,r+rec_size+1),max(0,c-rec_size):min(col,c+rec_size+1)])
    return output

def ShowMeans(means):
  """Show the cluster centers as images."""
  plt.figure() # HC: Removed '1' inside figure so it creates a new figure each time.
  plt.clf()
  for i in xrange(means.shape[0]):
    plt.subplot(1, means.shape[0], i+1)
    plt.imshow(means[i,:,:], cmap=plt.cm.gray)
  plt.draw()
  plt.show() # HC: Added show line
  # raw_input('Press Enter.')

def load_public_test():
    """ Loads the labeled data images, targets, and identities (sorted).
    """
    data = loadmat('public_test_images.mat')
    images = data['public_test_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    inputs = np.transpose(images, (0, 2, 1))
    inputs = inputs[:, :, ::-1]
    inputs = inputs.reshape(images.shape[0], images.shape[1]*images.shape[2])
    return inputs

def load_data_with_identity(include_mirror=False):
    """ Loads the labeled data images, targets, and identities (sorted).
    """
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

    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    targets = targets_sort[:,0]

    # Sort the images
    temp = np.append(images,identities,1)
    inputs_sort = temp[temp[:,-1].argsort()]
    inputs = inputs_sort[:,0:-1]

    # Return sorted identities:
    identities = inputs_sort[:,-1]

    return inputs, targets, identities

def load_data_with_identity_uniform(include_mirror=False):
    """ Loads the labeled data images, targets, and identities (sorted).
    Makes sure that the number of examples for each label is somewhat similar
    by deleting 4's and 7's target examples before returning. Keeps the first ~315 examples.
    """
    data = loadmat('labeled_images.mat')
    images = data['tr_images'].T # Transpose so the number of images is in first dimension: 2925, 32, 32
    targets = data['tr_labels']
    identities = data['tr_identity']

    # Get the images into the desired form: num_examples x 32 x 32 (and face is rotated in proper orientation)
    images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])

    # Sort the array based on the tr_labels
    # Sort the identities
    temp = np.append(identities,targets,1)
    identities_sort = temp[temp[:,1].argsort()]
    identities = identities_sort[:,0]

    # Sort the images
    temp = np.append(images,targets,1)
    images_sort = temp[temp[:,-1].argsort()]
    images = images_sort[:,0:-1]

    # Return sorted targets:
    targets = images_sort[:,-1]

    # Throw away some of the ones where the labels are 4 and 7
    indices_4 = np.where(targets == 4) # Index values where targets[indices_4] == 4
    start_index = 316 # Start index of where to delete the 4's
    end_index = indices_4[0][-1] + 1
    targets = np.delete(targets, indices_4[0][start_index:end_index])
    images = np.delete(images, indices_4[0][start_index:end_index], axis=0) # Delete the rows specified by indices_4[0][start:end]
    identities = np.delete(identities, indices_4[0][start_index:end_index])

    # Throw away some of the 7's
    indices_7 = np.where(targets == 7) # Index values where targets[indices_4] == 4
    start_index = 316 # Start index of where to delete the 4's
    end_index = indices_7[0][-1] + 1
    targets = np.delete(targets, indices_7[0][start_index:end_index])
    images = np.delete(images, indices_7[0][start_index:end_index], axis=0)
    identities = np.delete(identities, indices_7[0][start_index:end_index])

    print "Images shape: ", images.shape
    print "Targets shape: ", targets.shape
    print "identities shape: ", identities.shape

    # Generate a mirrored version if necessary:
    if include_mirror:
        # Unflatten the images
        images = images.reshape(images.shape[0], 32, 32)

        # Created mirrored instances
        mirrored_faces = images[:,:,::-1]
        images = np.append(images, mirrored_faces, 0)
        identities = np.append(identities, identities,0)
        targets = np.append(targets, targets,0)

        # Flatten the 32x32 to 1024 1D
        images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])

    return images, targets, identities

def reload_data_with_identity_normalized():
    """ Reloads the normalized data. Include mirror.
    """
    inputs = np.load('labeled_inputs_normalized.npy')
    targets = np.load('labeled_targets_normalized.npy')
    identities = np.load('labeled_identities_normalized.npy')

    return inputs, targets, identities

def load_data_with_identity_normalized(include_mirror=False):
    """ Loads the labeled data images, targets, and identities (sorted)
    and normalize the intensities of each image.
    """
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

    # Preprocess the data to normalize intensities
    for i in range(images.shape[0]):
        filt = np.array([[1,2,1],[2,4,2],[1,2,1]])
        gaussian = filt.astype(float)/filt.sum()
        gaussian_filter = signal.convolve2d(images[i,:,:], gaussian, boundary='symm', mode='same')
        std_filter = region_std(images[i,:,:],1)
        final = (images[i,:,:] - gaussian_filter).astype(float)/std_filter
        final = (((final/np.amax(final))+1)*128).astype(int)
        images[i,:,:] = final[:,:]
        print 'Done image ', i

    # Flatten the 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])

    # Sort the array based on the tr_identities
    # Sort the targets
    temp = np.append(targets,identities,1)
    targets_sort = temp[temp[:,1].argsort()]
    targets = targets_sort[:,0]

    # Sort the images
    temp = np.append(images,identities,1)
    inputs_sort = temp[temp[:,-1].argsort()]
    inputs = inputs_sort[:,0:-1]

    # Return sorted identities:
    identities = inputs_sort[:,-1]

    outfile = open('labeled_inputs_normalized.npy','w')
    np.save(outfile,inputs)
    outfile.close()

    outfile = open('labeled_targets_normalized.npy','w')
    np.save(outfile,targets)
    outfile.close()

    outfile = open('labeled_identities_normalized.npy','w')
    np.save(outfile,identities)
    outfile.close()

    return inputs, targets, identities


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

def reload_data_with_test_normalized():
    # Only load the original training data set that has intensity normalized
    train_inputs = np.load('train_inputs.npy')
    train_targets = np.load('train_targets.npy')
    valid_inputs = np.load('valid_inputs.npy')
    valid_targets = np.load('valid_targets.npy')
    test_inputs = np.load('test_inputs.npy')
    test_targets = np.load('test_targets.npy')

    return train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets


def load_data_with_test_normalized(include_mirror=False):
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

    # Preprocess the data to normalize intensities
    for i in range(images.shape[0]):
        # ShowMeans(images[i:(i+1)])
        # print images[i,:,:]
        # print "ith image shape: ",  images[i,:,:].shape
        filt = np.array([[1,2,1],[2,4,2],[1,2,1]])
        gaussian = filt.astype(float)/filt.sum()
        gaussian_filter = signal.convolve2d(images[i,:,:], gaussian, boundary='symm', mode='same')
        # print "Gaussian filter:"
        # print gaussian_filter
        std_filter = region_std(images[i,:,:],1)
        # print "std_filter:"
        # print std_filter
        final = (images[i,:,:] - gaussian_filter).astype(float)/std_filter
        # print "Final:"
        # print final
        # print 'final image shape: ', final.shape
        # print "Largest and smallest value in final:", np.amax(final), np.amin(final)
        final = (((final/np.amax(final))+1)*128).astype(int)
        images[i,:,:] = final[:,:]
        # print images[i,:,:]

        # # Output STD
        # output_std = np.zeros((1,32,32))
        # output_std[0,:,:] = std_filter[:,:]
        # # print output_std.shape
        # ShowMeans(output_std)

        # # Output gaussian:
        # output = np.zeros((1,32,32))
        # output[0,:,:] = gaussian_filter[:,:]
        # ShowMeans(output)

        # ShowMeans(images[i:(i+1)])

        print 'Done image ', i

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

def load_data_with_test_32x32(include_mirror=False):
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
    print data['tr_images'].T.shape[1], data['tr_images'].T.shape[2]
    images_sort = images_sort[:,0:-1].reshape(images.shape[0], data['tr_images'].T.shape[1], data['tr_images'].T.shape[2])
    valid_inputs = images_sort[0:num_unlabeled/2,:,:] # Tested that the -1 omits the identity last column
    test_inputs = images_sort[(num_unlabeled/2 + 1):num_unlabeled,:,:]
    train_inputs = images_sort[(num_unlabeled + 1):,:,:]
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

def load_unlabeled_data(include_mirror=False):
    data = loadmat('unlabeled_images.mat')
    images = data['unlabeled_images'].T

    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Flatten 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])

    return images

def load_unlabeled_data_normalized(include_mirror=False):
    data = loadmat('unlabeled_images.mat')
    images = data['unlabeled_images'].T

    if include_mirror:
        mirrored_faces = np.transpose(images, (0,2,1))
        rotated_faces = mirrored_faces[:,:,::-1]
        images = np.append(rotated_faces, mirrored_faces, 0)
    else:
        images = np.transpose(images, (0,2,1))[:,:,::-1]

    # Preprocess the data to normalize intensities
    for i in range(images.shape[0]):
        filt = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
        gaussian = filt.astype(float)/filt.sum()
        gaussian_filter = signal.convolve2d(images[i,:,:], gaussian, boundary='symm', mode='same')
        std_filter = region_std(images[i,:,:],2)
        final = (images[i,:,:] - gaussian_filter).astype(float)/std_filter
        final = (((final/np.amax(final))+1)*128).astype(int)
        images[i,:,:] = final[:,:]
        print 'Done image ', i

    # Flatten 32x32 to 1024 1D
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])

    outfile = open('unlabeled_normalized.npy','w')
    np.save(outfile,images)
    outfile.close()

    return images

def reload_unlabeled_data_normalized(include_mirror=False):
    # Just load from the data that's been pre-processed
    images = np.load('unlabeled_normalized.npy')
    return images



