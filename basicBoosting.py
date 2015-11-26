from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from utils import *

def run_AdaBoost():
    train_inputs, train_targets = load_train()
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, train_inputs, train_targets)
    print scores.mean()

def run_ExtremeRandFor():
    train_inputs, train_targets = load_train()
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_val_score(clf, train_inputs, train_targets)
    print scores.mean()

def run_ExtremeRandFor():
    train_inputs, train_targets = load_train()
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_val_score(clf, train_inputs, train_targets)
    print scores.mean()


if __name__ == '__main__':
    run_AdaBoost()
