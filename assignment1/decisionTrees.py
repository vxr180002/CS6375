# Written for CS6375.001 Spring 19, Assignment 1
# This program implements and test decision tree learning algorithm.
# Written by : Vishal Chandar Ramachandran
# Email: vxr180002

# how to run this script
# python3 decisiontrees.py <l> <k> training_set.csv validation_set.csv test_set.csv <to-print>
# l,k are integer values
# to-print : {yes, no} this parameter decides to print the tree

# report containing all accuracy before and after pruning is available in a file report.txt

import csv
import sys
import ast
from collections import Counter
import copy
import random
from math import log
import datetime

#--------------------------------------------- Defined class for tree nodes ------------------------------------------------

class decisionnode:

    def __init__(self, col = -1, value = None, results = None, tb = None, fb = None):

        
        self.col = col

        self.value = value

        self.results = results

        self.tb = tb

        self.fb = fb

#------------------------------------------- function to calculate the variance impurity of data set ---------------------------

def varianceImpurity(rows):

    if len(rows) == 0: return 0

    result = countClass(rows)

    total_samples = len(rows)

    variance_impurity = (result['0'] * result['1']) / (total_samples ** 2)

    return variance_impurity

#--------------------------------------------- function to calculate entropy of data set ----------------------------------------

def entropy(rows):

    log_base_2 = lambda x: log(x) / log(2)

    results = countClass(rows)

    Entropy = 0.0
    for r in results.keys():

        p = float(results[r]) / len(rows)

        Entropy = Entropy - p * log_base_2(p)

    return Entropy

#--------- function used to split data set based on entropy(default) or variance impurity after calculating gain -------------

def generateTree(rows, heuristicToUse=entropy):
	
    if len(rows) == 0: return decisionnode()
    
    class_entropy = heuristicToUse(rows)			# class column's entropy/InformtionGain
    
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1

    for col in range(0, column_count):
       
        global column_values

        column_values = {}

        for row in rows:

            column_values[row[col]] = 1
        
        for value in column_values.keys():
            
            (set1, set2) = divideDataSet(rows, col, value)
           
            p = float(len(set1)) / len(rows)

            gain = class_entropy - p * heuristicToUse(set1) - (1 - p) * heuristicToUse(set2)

            if gain > best_gain and len(set1) > 0 and len(set2) > 0: 

                best_gain = gain

                best_criteria = (col, value)

                best_sets = (set1, set2)

    if best_gain > 0:

        trueBranch = generateTree(best_sets[0])

        falseBranch = generateTree(best_sets[1])

        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)

    else:

        return decisionnode(results = countClass(rows))
	
#--------------------------------------- function for splitting dataset based on an attribute --------------------------------------

def divideDataSet(rows, column, value):

    split_data = None

    split_data = lambda row: row[column] == value

    set1 = []
    set2 = []
    
    for row in rows:
        if split_data(row):
            set1.append(row)
			
    for row in rows:
        if not split_data(row):
            set2.append(row)
	
    return (set1, set2)

#---------------------- count number of values based on class attribute (last column) and return a dictionary --------------------

def countClass(rows):

    results = {}

    for row in rows:

        r = row[len(row) - 1]

        if r not in results: results[r] = 0

        results[r] += 1

    return results


#-------------------------------------------------- print tree in required format -------------------------------------------

def displayTree(tree, header_data, indent):

    if tree.results != None:

        for key in tree.results:

            print(str(key))

    else:

        print("")

        print(indent + str(header_data[tree.col]) + ' = ' + str(tree.value) + ' : ', end="")

        displayTree(tree.tb, header_data, indent + '  |')

        print(indent + str(header_data[tree.col]) + ' = ' + str(int(tree.value) ^ 1) + ' : ', end="")

        displayTree(tree.fb, header_data, indent + '  |')

#--------------------------------------------------- function to calculate the accuracy -----------------------------------

def calculateTreeAccuracy(rows, tree):

    count_Of_correct_predictions = 0

    for row in rows:

        classified_value = classify(row, tree)

        if row[-1] == classified_value:

            count_Of_correct_predictions += 1

    accuracy = 100 * count_Of_correct_predictions / len(rows)

    return accuracy

#------------------------------------------ function to classify input data based on a learned tree ----------------------------

def classify(observation, tree):

    if tree.results != None:
        
        for key in tree.results:

            predicted_value = key

        return predicted_value

    else:

        v = observation[tree.col]
       
        if v == tree.value:

            branch = tree.tb

        else:

            branch = tree.fb

        predicted_value = classify(observation, branch)

    return predicted_value

#------------------------ function to count total number of non leaf nodes and label them according to number ------------------

def listNodes(nodes, tree, count):

    if tree.results != None:

        return nodes, count

    count += 1

    nodes[count] = tree

    (nodes, count) = listNodes(nodes, tree.tb, count)

    (nodes, count) = listNodes(nodes, tree.fb, count)

    return nodes, count

#------------------------------------- function to count number of target class -------------------------------------

def countOfClass(tree, class_occurence):

    if tree.results != None:
        
        for key in tree.results:

            class_occurence[key] += tree.results[key]

        return class_occurence



    left_branch_occurence = countOfClass(tree.fb, class_occurence)

    right_branch_occurence = countOfClass(tree.tb, left_branch_occurence)



    return right_branch_occurence

#--------------------------------------- replace subtree according to the pruning algorithm ----------------------------------

def findAndReplaceSubtree(tree_copy, subtree_to_replace, subtree_to_replace_with):

    if (tree_copy.results != None):

        return tree_copy

    if (tree_copy == subtree_to_replace):

        tree_copy = subtree_to_replace_with

        return tree_copy

    tree_copy.fb = findAndReplaceSubtree(tree_copy.fb, subtree_to_replace, subtree_to_replace_with)

    tree_copy.tb = findAndReplaceSubtree(tree_copy.tb, subtree_to_replace, subtree_to_replace_with)

    return tree_copy

#--------------------------------------------------- function to prune tree ---------------------------------------------

def pruneTree(tree, l, k, data):

    tree_best = tree

    best_accuracy = calculateTreeAccuracy(data, tree)

    tree_copy = None

    for i in range(1, l):

        m = random.randint(1, k)

        tree_copy = copy.deepcopy(tree)

        for j in range(1, m):

            (nodes, initial_count) = listNodes({}, tree_copy, 0)

            if (initial_count > 0):

                p = random.randint(1, initial_count)

                subtree_p = nodes[p]
               
                class_occurence = {'0': 0, '1': 0}

                count = countOfClass(subtree_p, class_occurence)
               
                if count['0'] > count['1']:

                    count['0'] = count['0'] + count['1']

                    count.pop('1')

                    subtree_p = decisionnode(results=count)

                else:

                    count['1'] = count['0'] + count['1']

                    count.pop('0')

                    subtree_p = decisionnode(results=count)

                tree_copy = findAndReplaceSubtree(tree_copy, nodes[p], subtree_p)
      
        curr_accuracy = calculateTreeAccuracy(data, tree_copy)

        if (curr_accuracy > best_accuracy):

            best_accuracy = curr_accuracy

            tree_best = tree_copy

    return tree_best, best_accuracy

# ------------------------------------------------- Main function: Program starts here ---------------------------------------------

if __name__ == "__main__":

    args = str(sys.argv)

    args = ast.literal_eval(args)

	# ast.literal_eval raises an exception if the input isn't a valid Python datatype, so the code won't be executed if it's not.
	
    if (len(args) < 6):

        print ("Input arguments should be 6. Please refer the Readme file regarding input format.")

    elif (args[3][-4:] != ".csv" or args[4][-4:] != ".csv" or args[5][-4:] != ".csv"):

        print(args[2])

        print ("Your training, validation and test file must be a .csv!")

    else:

        l = int(args[1])

        k = int(args[2])

        training_set = str(args[3])

        validation_set = str(args[4])

        test_set = str(args[5])

        to_print = str(args[6])

        with open(training_set, newline='', encoding='utf_8') as csvfile:

                csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')

                header_data = next(csvReader)
		
                train_training_data = list(csvReader)

        with open(validation_set, newline='', encoding='utf_8') as csvfile:

            csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')

            validation_training_data = list(csvReader)

        with open(test_set, newline='', encoding='utf_8') as csvfile:
            report_file = open('report.txt', 'a')
            report_file.write("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
            report_file.write("\nReport generated at "+str(datetime.datetime.now())+" :\n")

            csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')

            test_training_data = list(csvReader)

            l_array = [60, 15, 20, 25, 30, 35, 40, 45, 50, 55]

            k_array = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

            # ------------------------------ build tree using information gain heuristic -------------------------------------------

            generated_tree_IG = generateTree(train_training_data, heuristicToUse=entropy)

            print("Using Information Gain as a heuristic : \n")

            if(to_print.lower() == "yes"):

                print("\n Printing the learned tree : \n")

                displayTree(generated_tree_IG, header_data, '')

            train_accuracy = calculateTreeAccuracy(train_training_data, generated_tree_IG)
            
            print(" Training data accuracy : ", train_accuracy)
            report_file.write("\nTraining data accuracy : "+str(train_accuracy))

            validation_accuracy = calculateTreeAccuracy(validation_training_data, generated_tree_IG)

            print("\n Validation data accuracy : ", validation_accuracy)
            report_file.write("\nValidation data accuracy : "+str(validation_accuracy))

            test_accuracy = calculateTreeAccuracy(test_training_data, generated_tree_IG)

            print("\n Test data accuracy : ", test_accuracy)
            report_file.write("\nTest data accuracy : "+str(test_accuracy))

            (pruned_best_tree_validation, pruned_best_accuracy_validation) = pruneTree(generated_tree_IG, l, k,validation_training_data)

            print("\n Validation data accuracy after pruning : ", pruned_best_accuracy_validation)
            report_file.write("\nValidation data accuracy after pruning : "+str(pruned_best_accuracy_validation))

            (pruned_best_tree_test, pruned_best_accuracy_test) = pruneTree(generated_tree_IG, l, k, test_training_data)

            if (to_print.lower() == "yes"):

                print("\n Printing the pruned tree using test data : ")

                displayTree(pruned_best_tree_test, header_data, '')

            print("\n Test data accuracy after pruning : ", pruned_best_accuracy_test)
            report_file.write("\nTest data accuracy after pruning : "+str(pruned_best_accuracy_test))

            print("\n Calculating accuracy of test data with 10 combinations of l and k :")
            report_file.write("\nCalculating accuracy of test data with 10 combinations of l and k :")

            for l, k in  zip(l_array, k_array):

                (pruned_best_tree_test, pruned_best_accuracy_test) = pruneTree(generated_tree_IG, l, k,test_training_data)

                print("\n Test data accuracy after pruning with l = ", l," and k = " , k," : ", pruned_best_accuracy_test)
                report_file.write("\n\tTest data accuracy after pruning with l = "+str(l)+" and k = "+str(k)+" : "+str(pruned_best_accuracy_test))

            # --------------------------------------- build tree using variance impurity heuristic --------------------------------

            generated_tree_VI = generateTree(train_training_data, heuristicToUse = varianceImpurity)

            print("\nUsing Variance Impurity as a heuristic :\n")
            report_file.write("\n\nUsing Variance Impurity as a heuristic :\n")

            if (to_print.lower() == "yes"):

                print("\n Printing the learned tree : ")

                displayTree(generated_tree_VI, header_data, '')

            train_accuracy_VI = calculateTreeAccuracy(train_training_data, generated_tree_VI)

            print("\n Training data accuracy : ", train_accuracy_VI)
            report_file.write("\nTraining data accuracy : "+str(train_accuracy_VI))

            validation_accuracy_VI = calculateTreeAccuracy(validation_training_data, generated_tree_VI)

            print("\n Validation data accuracy : ", validation_accuracy_VI)
            report_file.write("\nValidation data accuracy : "+str(validation_accuracy_VI))

            test_accuracy_VI = calculateTreeAccuracy(test_training_data, generated_tree_VI)

            print("\n Test data accuracy : ", test_accuracy_VI)
            report_file.write("\nTest data accuracy : "+str(test_accuracy_VI))

            (pruned_best_tree_validation_VI, pruned_best_accuracy_validation_VI) = pruneTree(generated_tree_VI, l, k, validation_training_data)

            print("\n Validation data accuracy after pruning: ", pruned_best_accuracy_validation_VI)
            report_file.write("\nValidation data accuracy after pruning: "+str(pruned_best_accuracy_validation_VI))

            (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = pruneTree(generated_tree_VI, l, k, test_training_data)

            if (to_print.lower() == "yes"):

                print("\n Printing the pruned tree using on test data : ")

                displayTree(pruned_best_tree_test_VI, header_data, '')

            print("\n Test data accuracy after pruning : ", pruned_best_accuracy_test_VI)
            report_file.write("\nTest data accuracy after pruning : "+str(pruned_best_accuracy_test_VI))

            print("\n Calculating accuracies of test data with 10 combinations of l and k :")
            report_file.write("\nCalculating accuracies of test data with 10 combinations of l and k :")

            for l, k in zip(l_array, k_array):

                (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = pruneTree(generated_tree_VI, l, k,test_training_data)

                print("\n Test data accuracy after pruning with l = ", l, " and k = ", k ," : ", pruned_best_accuracy_test_VI)
                report_file.write("\n\tTest data accuracy after pruning with l = "+str(l)+" and k = "+str(k)+" : "+str(pruned_best_accuracy_test_VI))

            report_file.write("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            report_file.close()