import math

import naive_bayes
import util

data_set = util.load_data_set('datasets/Compound.csv')

if __name__ == "__main__":

    # compare & visualize between dataset and evaluated data
    util.visualize(data_set)

    # Make Naive Bayes Classifier (Gaussian)
    classifier = naive_bayes.NaiveBayes(data_set)

    # Evaluate date set & make confusion matrix
    evaluated_data, confusion_matrix = classifier.evaluate(data_set)

    # compare & visualize between dataset and evaluated data
    util.compare_data(data_set, evaluated_data)

    # performance calculation (accuracy)
    # print "Accuracy : {}".format(util.performance_calculation(confusion_matrix))

    # performance calculation (f1_score)
    util.performance_calculation(confusion_matrix, mode="f1_micro_average")

    #
    util.decision_boundary(data_set, classifier)
