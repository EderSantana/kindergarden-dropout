import numpy as np
import pylab
from scipy.io import loadmat
#from sparseFiltering import *
from PIL import Image

def displayData(X, save_path, example_width = False, display_cols = False):
    """
    Display 2D data in a nice grid
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.
    """
    # compute rows, cols
    m,n = X.shape
    if not example_width:
        example_width = int(np.round(np.sqrt(n)))
    example_height = (n/example_width)
    # Compute number of items to display
    if not display_cols:
        display_cols = int(np.sqrt(m))
    display_rows = int(np.ceil(m/display_cols))
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+display_rows * (example_height+pad),
        pad+display_cols * (example_width+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex>=m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = abs(X[curr_ex,:]).max()
            i_inds = example_width*[pad+j * (example_height+pad)+q for q in range(example_height)]
            j_inds = [pad+i * (example_width+pad)+q
                        for q in range(example_width)
                        for nn in range(example_height)]
            try:
                newData = (X[curr_ex,:].reshape((example_height,example_width))).T/max_val
            except:
                print X[curr_ex,:].shape
                print (example_height,example_width)
                raise
            display_array[i_inds,j_inds] = newData.flatten()
            curr_ex+=1
        if curr_ex>=m:
            break
    # Display the image
    visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(save_path)
    return display_array

def displayMultiChannel(X, save_path, example_width = False, display_cols = False):
    """
    Display Multi-Channel filters in a nice grid
    Each channel goes in one column
    Input filters are expected in the following configuration batches x channels x row x columns
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.
    """
    # compute rows, cols
    b,c,m,n = X.shape
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+b * (m+pad),
        pad+c * (n+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(b):
        for i in range(c):
            # Copy the patch
            # Get the max value of the patch
            display_array[j*(m+pad)+pad:(j+1)*(m+pad), (i*(n+pad))+pad:((i+1)*(n+pad))] = X[j,i,:,:]
    # Display the image
    visual = display_array #(display_array - display_array.min()) / (display_array.max() - display_array.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(save_path)
    return display_array

def displaySubplotChannel(X, save_path, example_width = False, display_cols = False):
    """
    Display Multi-Channel filters in a nice grid
    Each channel goes in one column
    Input filters are expected in the following configuration batches x channels x row x columns
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.
    """
    # compute rows, cols
    b,c,m,n = X.shape
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+b * (m+pad),
        pad+c * (n+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(b):
        for i in range(c):
            # Copy the patch
            # Get the max value of the patch
            display_array[j*(m+pad)+pad:(j+1)*(m+pad), (i*(n+pad))+pad:((i+1)*(n+pad))] = X[j,i,:,:]
            
    # Display the image
    visual = display_array #(display_array - display_array.min()) / (display_array.max() - display_array.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(save_path)
    return display_array

def displayMultiChannel(X, save_path, example_width = False, display_cols = False):
    """
    Display Multi-Channel filters in a nice grid
    Each channel goes in one column
    Input filters are expected in the following configuration batches x channels x row x columns
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.
    """
    # compute rows, cols
    b,c,m,n = X.shape
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+b * (m+pad),
        pad+c * (n+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(b):
        for i in range(c):
            # Copy the patch
            # Get the max value of the patch
            display_array[j*(m+pad)+pad:(j+1)*(m+pad), (i*(n+pad))+pad:((i+1)*(n+pad))] = X[j,i,:,:]
    # Display the image
    visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(save_path)
    return display_array


def displaySubplotChannel(X, save_path, example_width = False, display_cols = False):
    """
    Display Multi-Channel filters in a nice grid
    Each channel goes in one column
    Input filters are expected in the following configuration batches x channels x row x columns
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.
    """
    # compute rows, cols
    b,c,m,n = X.shape
    pad = 1
    # Setup blank display
    display_array = -np.zeros((pad+b * (m+pad),
        pad+c * (n+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(b):
        for i in range(c):
            # Copy the patch
            # Get the max value of the patch
            display_array[j*(m+pad)+pad:(j+1)*(m+pad), (i*(n+pad))+pad:((i+1)*(n+pad))] = X[j,i,:,:]
            
            pylab.subplot(b,c,curr_ex)
            curr_ex += 1
            img = X[j,i,:,:]#display_array[j*(m+pad)+pad:(j+1)*(m+pad)] 
            pylab.imshow(img,cmap='gray')#imshow((img - img.min())/(img.max()-img.min()),cmap='gray')
            pylab.xticks([])
            pylab.yticks([])
            pylab.subplots_adjust(wspace=0.01,hspace=.1)
    # Display the image
    visual = display_array #(display_array - display_array.min()) / (display_array.max() - display_array.min())
    result = Image.fromarray((visual * 255).astype(np.uint8))
    result.save(save_path)
    pylab.show()
    return display_array


