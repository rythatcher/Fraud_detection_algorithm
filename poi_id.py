import sys
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

sys.path.append("../tools/")

features_list = ['poi','salary', 'total_stock_value', 'from_poi_ratio', \
 'expenses'] 

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Removing outlier and adding custom features
data_dict.pop('TOTAL')

for x in data_dict:
    data_dict[x]['from_poi_ratio'] = str(float(data_dict[x]\
    ['from_poi_to_this_person'])/float(data_dict[x]['to_messages']))
    data_dict[x]['to_poi_ratio'] = str(float(data_dict[x]\
    ['from_this_person_to_poi'])/float(data_dict[x]['from_messages']))
# to_poi_ratio removed from fit due to data leakage issue
    if data_dict[x]['from_poi_ratio'] == 'nan':
        data_dict[x]['from_poi_ratio'] = 'NaN'
    if data_dict[x]['to_poi_ratio'] == 'nan':
        data_dict[x]['to_poi_ratio'] = 'NaN'
        
my_dataset = data_dict

# Conversion of dictionary to features/labels numpy arrays
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split = 10, class_weight = 'balanced')

#==============================================================================
#  Original tuning of decision tree 
# parameters = {'min_samples_split' : [1, 2, 5, 10, 20, 30, 40], \
# class_weight' : [None, 'balanced']}
# from sklearn.grid_search import GridSearchCV
# clf = GridSearchCV(clf1, parameters, scoring = 'f1')
#==============================================================================

t0 = time()    
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0

kf = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train, test in kf:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for x in train:
        features_train.append(features[x])
        labels_train.append(labels[x])
    for x in test:
        features_test.append(features[x])
        labels_test.append(labels[x])
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            break

total_predictions = true_negatives + false_negatives + false_positives + \
true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)
f1 = 2.0 * true_positives/(2*true_positives + false_positives+\
false_negatives)

# Printing all metrics from the prediction
print "Accuracy:", round(accuracy, 4), 'Precision:', round(precision, 4), \
"Recall:", round(recall, 4), "f1:", round(f1, 4)

# Uncomment to print feature importances from decision tree
#==============================================================================
# feature_importances = clf.feature_importances_
# for x in range(0, len(features_list)-1):
#     print features_list[x+1], ':', feature_importances[x]
#==============================================================================
print "Fitting time:", round(time() - t0, 3), "s"

# Output classifier, dataset and features for external validation 
dump_classifier_and_data(clf, my_dataset, features_list)




