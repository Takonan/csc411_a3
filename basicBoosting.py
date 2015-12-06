from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LabelKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.scorer import check_scoring
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from utils import *
import time

def run_AdaBoost(num_estimator=10, num_iter=5, include_mirror=False, do_cv=False):
    train_inputs, train_targets, valid_inputs, valid_targets = load_data(include_mirror)
    myClassifier = LogisticRegression()
    # myClassifier = Perceptron(n_iter=num_iter)
    # myClassifier = SGDClassifier(loss='perceptron',n_iter=num_iter)
    clf = AdaBoostClassifier(base_estimator=myClassifier, n_estimators=num_estimator)

    if do_cv:
        # Do cross validation
        scores = cross_val_score(clf, train_inputs, train_targets)
        return scores.mean()

    else:
        # Do just one validation
        clf.fit(train_inputs, train_targets)
        pred = clf.predict(valid_inputs)
        score = (pred == valid_targets).mean()
        return score

    # clf = AdaBoostClassifier(n_estimators=100)
    # scores = cross_val_score(clf, train_inputs, train_targets, n_jobs=-1)
    # print scores.mean()

def run_ExtremeRandFor(include_mirror=False):
    train_inputs, train_targets, valid_inputs, valid_targets = load_data(include_mirror)
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, n_jobs=-1)
    scores = cross_val_score(clf, train_inputs, train_targets, n_jobs=-1)
    print scores.mean()

def run_RandFor(include_mirror=False):
    train_inputs, train_targets, valid_inputs, valid_targets = load_data(include_mirror)
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, n_jobs=-1)
    scores = cross_val_score(clf, train_inputs, train_targets, n_jobs=-1)
    print scores.mean()

def run_Bagging(num_estimator=10, num_iter=5, include_mirror=False, do_cv=False, reload=False):
    if not reload:
        train_inputs, train_targets, valid_inputs, valid_targets = load_data(include_mirror)
    else:
        train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = reload_data_with_test_normalized()
    # myClassifier = LinearSVC()
    # myClassifier = RidgeClassifier()
    # myClassifier = Perceptron(n_iter=num_iter)
    # myClassifier = SGDClassifier(loss='perceptron',n_iter=num_iter)
    myClassifier = OneVsRestClassifier(LinearSVC(random_state=0))
    # clf = BaggingClassifier(base_estimator=myClassifier, n_estimators=num_estimator, n_jobs=-1)

    if do_cv:
        # Do cross validation
        scores = cross_val_score(clf, train_inputs, train_targets)
        return scores.mean()

    else:
        # Do just one validation
        # clf.fit(train_inputs, train_targets)
        pred = myClassifier.fit(train_inputs, train_targets).predict(valid_inputs)
        # pred = clf.predict(valid_inputs)
        score = (pred == (valid_targets)).mean()
        return score

    return

def pca_SVM(normalized_intensity=False, ratio=0.25):
    if not normalized_intensity:
        # Perform PCA on the unlabeled data (Not include the mirror)
        images = load_unlabeled_data()

        # start = time.time()
        # pca = PCA(n_components=images.shape[1]*ratio)
        # unlabeled_pca = pca.fit_transform(images)
        # elasped = time.time() - start
        # print "Done doing PCA fit with ratio %f" % (ratio)
        # print "It took %f seconds" % elasped

        # Now do Kernel PCA on the unlabeled_pca
        kpca = KernelPCA(kernel="rbf", gamma=15)
        start = time.time()
        # unlabeled_kpca = kpca.fit(unlabeled_pca)
        unlabeled_kpca = kpca.fit(images[0:6000])
        elasped = time.time() - start
        print "Done Kernel PCA fit"
        print "It took %f seconds" % elasped

        # # Perform SVM on the PCA transformed data
        # train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets = load_data_with_test(True)

        # train_inputs = pca.transform(train_inputs)
        # valid_inputs = pca.transform(valid_inputs)
        # test_inputs = pca.transform(test_inputs)

        # Train one vs one SVM's
        clf = SVC()
        # clf.fit(train_inputs, train_targets)

        # val_pred = clf.predict(valid_inputs)
        # print valid_targets
        # print val_pred
        # print accuracy_score(valid_targets, val_pred)
        # print(classification_report(valid_targets, val_pred))

        # test_pred = clf.predict(test_inputs)
        # print test_targets
        # print test_pred
        # print accuracy_score(test_targets, test_pred)
        # print(classification_report(test_targets, test_pred))

        inputs, targets, identities = load_data_with_identity(True)
        inputs = kpca.transform(inputs)
        print "Dimension of inputs:", inputs.shape
        lkf = LabelKFold(identities, n_folds=3)
        # for train_index, test_index in lkf:
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     X_train, X_test = inputs[train_index], inputs[test_index]
        #     y_train, y_test = targets[train_index], targets[test_index]


        # # Do not legit cross validation:
        scores = cross_val_score(clf, inputs, targets, cv=lkf, n_jobs=-1)
        print scores.mean()

    return

if __name__ == '__main__':
    # print "Running classification algorithms with original training data set:"
    # # run_AdaBoost()
    # # run_ExtremeRandFor()
    # # run_RandFor()
    # start = time.time()
    # run_Bagging()
    # elasped = time.time() - start
    # print "Elasped time: ", elasped
    # print "Running classification algorithms with original training data set and mirrorred data set:"
    # # run_AdaBoost(True)
    # # run_ExtremeRandFor(True)
    # # run_RandFor(True)
    # start = time.time()
    # run_Bagging(True)
    # elasped = time.time() - start
    # print "Elasped time: ", elasped


    # for num_estimator in [1]: #[10, 25, 50]:
    #     for num_iter in [15]: #[5, 10, 25, 50]:
    #         # print "Original Set, num_estimator: %d, num_iter: %d, accuracy: %f" % (num_estimator, num_iter, run_Bagging(num_estimator, num_iter, False, False))
    #         print "Original + Mirrored Set, num_estimator: %d, num_iter: %d, accuracy: %f" % (num_estimator, num_iter, run_Bagging(num_estimator, num_iter, True, False, True))

    pca_SVM()
