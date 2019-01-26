"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
import numpy as np
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        # print(Counter(y).most_common(1))
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        num0 = Counter(y).most_common(2)[0][1]
        num1 = Counter(y).most_common(2)[1][1]
        prob0 = num0 / float(num0 + num1)
        prob1 = num1 / float(num0 + num1)
        prob_dict = {
            0.0: prob0,
            1.0: prob1
        }
        self.probabilities_ = prob_dict
        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n,d = X.shape
        class_arr = [0.0,1.0]
        prob_arr = [self.probabilities_[0.0], self.probabilities_[1.0]]
        y = np.random.choice(class_arr,n, prob_arr)   
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error_sum = 0
    test_error_sum = 0

    for num_trial in range(ntrials):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=num_trial)
        
        # training error
        clf.fit(Xtrain, ytrain)
        y_pred_train = clf.predict(Xtrain)
        temp_train_error = 1 - metrics.accuracy_score(ytrain, y_pred_train, normalize=True)
        train_error_sum += temp_train_error

        # test error
        clf.fit(Xtest, ytest)
        y_pred_test = clf.predict(Xtrain)
        temp_test_error = 1 - metrics.accuracy_score(ytrain, y_pred_test, normalize=True)
        test_error_sum += temp_test_error

    train_error = train_error_sum / 100
    test_error = test_error_sum / 100
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print 'Plotting...'
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    rand_clf = RandomClassifier()
    rand_clf.fit(X, y)
    rand_y_pred = rand_clf.predict(X)
    rand_train_error = 1 - metrics.accuracy_score(y, rand_y_pred, normalize=True)
    print '\t-- training error: %.3f' % rand_train_error
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    dt_clf = DecisionTreeClassifier("entropy")
    dt_clf.fit(X,y)
    dt_y_pred = dt_clf.predict(X)
    dt_train_error = 1 - metrics.accuracy_score(y, dt_y_pred, normalize=True)
    print '\t-- training error: %.3f' % dt_train_error
    
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    # save the classifier -- requires GraphViz and pydot
    # import StringIO, pydot
    # from sklearn import tree
    # dot_data = StringIO.StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                      feature_names=Xnames)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("dtree.pdf") 

    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    mv_train_error_cv, mv_test_error_cv = error(MajorityVoteClassifier(), X, y)
    rand_train_error_cv, rand_test_error_cv = error(RandomClassifier(), X, y)
    dt_train_error_cv, dt_test_error_cv = error(DecisionTreeClassifier(), X, y)
    print '\t-- MV training error: %.3f' % mv_train_error_cv + ', test error: %.3f' % mv_test_error_cv
    print '\t-- RAND training error: %.3f' % rand_train_error_cv + ', test error: %.3f' % rand_test_error_cv
    print '\t-- DT training error: %.3f' % dt_train_error_cv + ', test error: %.3f' % dt_test_error_cv

    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    train_arr = []
    test_arr = []
    mv_test_arr = []
    rand_test_arr = []
    depth_arr = []

    for i in range(1,21):
        depth_clf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        depth_train_error, depth_test_error = error(depth_clf, X, y)
        train_arr.append(depth_train_error)
        test_arr.append(depth_test_error)

        mv_train_error, mv_test_error = error(MajorityVoteClassifier(), X, y)
        mv_test_arr.append(mv_test_error)

        rand_train_error, rand_test_error = error(RandomClassifier(), X, y)
        rand_test_arr.append(rand_test_error)

        depth_arr.append(i)

    plt.plot(depth_arr, train_arr, label="Avg Training Error")
    plt.plot(depth_arr, test_arr, label="Avg Test Error")
    plt.plot(depth_arr, mv_test_arr, label="MajorityVote Test Error")
    plt.plot(depth_arr, rand_test_arr, label="Random Test Error")
    plt.legend(loc=0, fontsize="x-small")
    plt.xlabel("Depth Limit")
    plt.ylabel("Error")
    plt.xticks(range(1,21))
    plt.title("Error vs. Depth Limit")
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    lc_train_arr = []
    lc_test_arr = []
    lc_mv_arr = []
    lc_rand_arr = []
    split_arr = []

    for j in [float(k) / 20 for k in range(1, 20)]:
        lc_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        lc_train_error, lc_test_error = error(lc_clf, X, y, test_size=j)
        lc_train_arr.append(lc_train_error)
        lc_test_arr.append(lc_test_error)

        lc_mv_train_error, lc_mv_test_error = error(MajorityVoteClassifier(), X, y, test_size=j)
        lc_mv_arr.append(lc_mv_test_error)

        lc_rand_train_error, lc_rand_test_error = error(RandomClassifier(), X, y, test_size=j)
        lc_rand_arr.append(lc_rand_test_error)

        split_arr.append(j)

    plt.plot(split_arr, lc_train_arr, label="Avg Training Error")
    plt.plot(split_arr, lc_test_arr, label="Avg Test Error")
    plt.plot(split_arr, lc_mv_arr, label="MajorityVote Test Error")
    plt.plot(split_arr, lc_rand_arr, label="Random Test Error")
    plt.legend(loc=0, fontsize="x-small")
    plt.xlabel("Training Data Split Size")
    plt.ylabel("Error")
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.title("Error vs. Split Size")
    plt.show()


    
    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
