import os
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import FeatureExtractor as extractor


def run():
    ''' Establish directory paths for training and validation data '''
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sunny_train_dirpath = ROOT_DIR + '/data/c_sunny/'
    overcast_train_dirpath = ROOT_DIR + '/data/c_overcast/'
    sunny_test_dirpath = ROOT_DIR + '/data/3000_images_test/c_sunny/'
    overcast_test_dirpath = ROOT_DIR + '/data/3000_images_test/c_overcast/'

    ''' Perform feature extraction on training data '''
    sunny_train_c = [f for f in os.listdir(sunny_train_dirpath)]
    overcast_train_c = [f for f in os.listdir(overcast_train_dirpath)]
    train_feats = extractor.get_hog_hist_features(sunny_train_c, overcast_train_c, sunny_train_dirpath, overcast_train_dirpath)

    ''' Perform feature extraction on validation data '''
    sunny_test_c = [f for f in os.listdir(sunny_test_dirpath)]
    overcast_test_c = [f for f in os.listdir(overcast_test_dirpath)]
    test_feats = extractor.get_hog_hist_features(sunny_test_c, overcast_test_c, sunny_test_dirpath, overcast_test_dirpath)

    ''' Naive Bayes Classifier '''
    train_feats = shuffle(train_feats)
    test_feats = shuffle(test_feats)
    X_train = train_feats.iloc[:, :len(train_feats.columns) - 1]
    y_train = train_feats['label']
    X_test = test_feats.iloc[:, :len(test_feats.columns) - 1]
    y_test = test_feats['label']
    clf_nb = GaussianNB()
    model = clf_nb.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred, normalize=True)
    mat = confusion_matrix(y_test, y_pred)
    print(mat)
    print("Number of mislabeled points out of a total {0} points : {1}".format(len(X_test), (y_test != y_pred).sum()))
    print("Naive Bayes Classifier test accuracy = ", acc_score)


def plotImages(sunny_dirpath, overcast_dirpath):
    extractor.grayscale_feature_plot(sunny_dirpath, overcast_dirpath)
    extractor.HOG_feature_plot(sunny_dirpath, overcast_dirpath)

if __name__ == '__main__':
    run()
