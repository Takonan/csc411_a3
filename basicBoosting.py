from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.scorer import check_scoring
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

def run_Bagging(num_estimator=10, num_iter=5, include_mirror=False, do_cv=False):
    train_inputs, train_targets, valid_inputs, valid_targets = load_data(include_mirror)
    # myClassifier = LinearSVC()
    # myClassifier = RidgeClassifier()
    # myClassifier = Perceptron(n_iter=num_iter)
    myClassifier = SGDClassifier(loss='perceptron',n_iter=num_iter)
    clf = BaggingClassifier(base_estimator=myClassifier, n_estimators=num_estimator, n_jobs=-1)

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


    for num_estimator in [25]: #[10, 25, 50]:
        for num_iter in [5]: #[5, 10, 25, 50]:
            print "Original Set, num_estimator: %d, num_iter: %d, accuracy: %f" % (num_estimator, num_iter, run_AdaBoost(num_estimator, num_iter, False, False))
            print "Original + Mirrored Set, num_estimator: %d, num_iter: %d, accuracy: %f" % (num_estimator, num_iter, run_AdaBoost(num_estimator, num_iter, True, False))
