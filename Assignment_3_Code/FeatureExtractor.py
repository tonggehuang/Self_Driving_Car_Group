import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import data, exposure
from skimage.feature import hog


def resize_cvt2gray(inputpath):
    img = cv2.imread(inputpath)
    img_resize = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    return img_gray


def get_hog_hist_features(sunny_c, overcast_c, sunny_dirpath, overcast_dirpath):
    ''' sunny data '''
    hist_sunny = []
    hog_sunny = []
    for image in sunny_c:
        img_gray = resize_cvt2gray(sunny_dirpath + image)
        hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
        fd_sunny = hog(img_gray, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2')

        hist_sunny.append(hist.ravel())
        hog_sunny.append(fd_sunny)

        df_sunny_hist = pd.DataFrame(np.array(hist_sunny))
        df_sunny_hog = pd.DataFrame(np.array(hog_sunny))

        df_sunny = pd.DataFrame(np.hstack((df_sunny_hist, df_sunny_hog)))
        df_sunny['label'] = 1

    ''' overcast data '''
    hist_overcast = []
    hog_overcast = []
    for image in overcast_c:
        img_gray = resize_cvt2gray(overcast_dirpath + image)
        hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
        fd_overcast = hog(img_gray, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2')

        hist_overcast.append(hist.ravel())
        hog_overcast.append(fd_overcast)

        df_overcast_hist = pd.DataFrame(np.array(hist_overcast))
        df_overcast_hog = pd.DataFrame(np.array(hog_overcast))

        df_overcast = pd.DataFrame(np.hstack((df_overcast_hist, df_overcast_hog)))
        df_overcast['label'] = 0

    df_final = pd.concat([df_overcast, df_sunny], ignore_index=True)
    return df_final


def grayscale_feature_plot(sunny_dirpath, overcast_dirpath):
    ''' Raw image and hist visualization - only using 1 example image '''
    img_sunny = cv2.imread(sunny_dirpath + 'stereo_centre_1403619592348819.png')
    img_overcast = cv2.imread(overcast_dirpath + 'stereo_centre_1403773067908582.png')
    img_sunny = cv2.resize(img_sunny, (128, 128))
    img_overcast = cv2.resize(img_overcast, (128, 128))
    img_sunny_rgb = cv2.cvtColor(img_sunny, cv2.COLOR_BGR2RGB)
    img_overcast_rgb = cv2.cvtColor(img_overcast, cv2.COLOR_BGR2RGB)
    img_sunny_gray = cv2.cvtColor(img_sunny, cv2.COLOR_BGR2GRAY)
    img_overcast_gray = cv2.cvtColor(img_overcast, cv2.COLOR_BGR2GRAY)

    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[0,2])
    ax3 = plt.subplot(gs[1,0])
    ax4 = plt.subplot(gs[1,1])
    ax5 = plt.subplot(gs[1,2])
    # sunny
    ax0.imshow(img_sunny_rgb)
    ax0.set_title('img_sunny_rgb')
    ax1.imshow(img_sunny_gray,cmap='gray')
    ax1.set_title('img_sunny_gray')
    ax2.hist(img_sunny_gray.ravel(),256,[0,256])
    ax2.set_title('hist_sunny_gray')
    ax2.set_ylim([0,500])
    # overcast
    ax3.imshow(img_overcast_rgb)
    ax3.set_title('img_overcast_rgb')
    ax4.imshow(img_overcast_gray,cmap='gray')
    ax4.set_title('img_overcast_gray')
    ax5.hist(img_overcast_gray.ravel(),256,[0,256])
    ax5.set_title('hist_overcast_gray')
    ax5.set_ylim([0,500])
    fig.show()

def HOG_feature_plot(sunny_dirpath, overcast_dirpath):
    ''' Raw image and hist visualization - only using 1 example image '''
    img_sunny = cv2.imread(sunny_dirpath + 'stereo_centre_1403619592348819.png')
    img_overcast = cv2.imread(overcast_dirpath + 'stereo_centre_1403773067908582.png')
    img_sunny = cv2.resize(img_sunny, (128, 128))
    img_overcast = cv2.resize(img_overcast, (128, 128))
    img_sunny_rgb = cv2.cvtColor(img_sunny, cv2.COLOR_BGR2RGB)
    img_overcast_rgb = cv2.cvtColor(img_overcast, cv2.COLOR_BGR2RGB)
    img_sunny_gray = cv2.cvtColor(img_sunny, cv2.COLOR_BGR2GRAY)
    img_overcast_gray = cv2.cvtColor(img_overcast, cv2.COLOR_BGR2GRAY)

    ''' sunny '''
    fd_sunny, hog_image_sunny = hog(img_sunny_gray, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                    block_norm='L2', visualise=True)
    hog_rescale_sunny = exposure.rescale_intensity(hog_image_sunny, in_range=(0, 10))

    ''' overcast '''
    fd_overcast, hog_image_overcast = hog(img_overcast_gray, orientations=6, pixels_per_cell=(16, 16),
                                          cells_per_block=(1, 1), block_norm='L2', visualise=True)
    hog_rescale_overcast = exposure.rescale_intensity(hog_image_overcast, in_range=(0, 10))

    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[0,2])
    ax3 = plt.subplot(gs[1,0])
    ax4 = plt.subplot(gs[1,1])
    ax5 = plt.subplot(gs[1,2])
    # sunny
    ax0.imshow(img_sunny_rgb)
    ax0.set_title('img_sunny_rgb')
    ax1.imshow(img_sunny_gray, cmap=plt.cm.gray)
    ax1.set_title('img_sunny_rgb')
    ax2.imshow(hog_rescale_sunny, cmap=plt.cm.gray)
    ax2.set_title('HOG_sunny')

    # overcast
    ax3.imshow(img_overcast_rgb)
    ax3.set_title('img_overcast_rgb')
    ax4.imshow(img_overcast_gray, cmap=plt.cm.gray)
    ax4.set_title('img_overcast_gray')
    #ax3.axis('off')
    ax5.imshow(hog_rescale_overcast, cmap=plt.cm.gray)
    ax5.set_title('HOG_overcast')
    fig.show()