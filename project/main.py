import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

training_images = []     # half of pictures of each object
testing_images = []      # second half, remaining pictures
edges_training = []      # analogic variables, but for edges for particular images
edges_testing = []
Hu_training = []         # analogic variables, but for hu's invariant moments
Hu_testing = []
training_images_ids = [] # variable for splitting images into rightful sets
testing_images_ids = []
X_test_2 = []            # temporary variable for testing stage
y_test_2 = []
score_clf = []           # array for holding scores for each single classification
mod = 0                  # modifier, responsible for navigating between objects in feature extraction part
nr_array = np.arange(0, 356, 5) # array with numbers for randomizing images in both sets
test_array = np.arange(0, 3601, 36) # array arranged for picking testing classes for svm
obj = 'obj'              # variable responsible for navigation between objects' images in images loading stage
                         # Path to folder containing objects' images at local repository
path = 'C:/Users/Veteran/Object-recognition-using-SVM-models/coil-100/'

# Feature extraction from images
for i in range(1,101,1):        # loop over particular objects' folders
    obj_nr = obj + str(i)       # storing object's number (label)

    # Creating two sets of earlier separated images
    #np.random.shuffle(nr_array) # randomizing images of particular object
    #training_images_ids = nr_array[:len(nr_array)//2] # Randomization technique
    #testing_images_ids = nr_array[len(nr_array)//2:] 

    for k in range(72):         # Selecting picture every 10 degrees for training, and rest for testing
        if k % 2 == 0:
            training_images_ids.append(nr_array[k])
        else:
            testing_images_ids.append(nr_array[k])

    for k in range(36):         # loop over images
        # Building paths according to earlier established sets
        objpath_train = path + obj_nr + '__' + str(training_images_ids[k]) + '.png'
        objpath_test = path + obj_nr + '__' + str(testing_images_ids[k]) + '.png'

        # Loading images into separate sets (honouring labels - numbers of objects)
        training_image = cv2.imread(objpath_train, 0)
        training_images.append([training_image, i])
        testing_image = cv2.imread(objpath_test, 0)
        testing_images.append([testing_image, i])

    # Calculating edges for each image
    for k in range(36):
        edges_training.append([cv2.Canny(training_images[k+mod][0],100,200), i])
        edges_testing.append([cv2.Canny(testing_images[k+mod][0],100,200), i])

    # Calculating Hu's invariant values for each image
    for k in range(36):
        temp = np.array(cv2.HuMoments(cv2.moments(edges_training[k+mod][0])).flatten())
        temp = np.append(temp, [i])
        Hu_training.append(temp)
        temp = np.array(cv2.HuMoments(cv2.moments(edges_testing[k+mod][0])).flatten())
        temp = np.append(temp, [i])
        Hu_testing.append(temp)
    
    mod+=36                   # Jumping to next object

# Classification part
Hu_training = np.array(Hu_training)
Hu_testing = np.array(Hu_testing)

X_train = Hu_training[:,:-1]  # Splitting training set into atributes and labels
y_train = Hu_training[:,-1]

X_test = Hu_testing[:,:-1]    # Splitting testing set into atributes and labels
y_test = Hu_testing[:,-1]

clf = SVC()                   # Initializing classifier
clf.fit(X_train, y_train)     # Training classifier and calculating the score on test set

for j in range(0, 100, 1):    # Testing stage - first image is being comparised to every other from testing set
    X_test_2.clear()
    y_test_2.clear()
    for k in range(0, 36, 1):
        X_test_2.append(X_test[k])
        y_test_2.append(y_test[k])

    for l in range(test_array[j], test_array[j]+36, 1):
        X_test_2.append(X_test[l])
        y_test_2.append(y_test[l])

    score = accuracy_score(y_test_2, clf.predict(X_test_2)) 
    score_clf.append(score)  # Calculating accuracy for each classification and saving it in array

#for k in range(72): # For displaying images
    #cv2.imshow('', edges_training[k][0])
    #cv2.waitKey()

print('Classification accuracy for SVM: ', score_clf)