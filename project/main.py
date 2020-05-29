import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

training_images = []     # half of pictures of each object
testing_images = []      # second half, remaining pictures
edges_training = []      # analogic variables, but for edges for particular images
edges_testing = []
Hu_training = []        # analogic variables, but for hu's invariant moments
Hu_testing = []
nr_array = np.arange(72) # array with numbers for randomizing images in both sets
obj = 'obj'              # variable responsible for navigation between objects' images
                         # Path to folder containing objects' images at local repository
path = 'C:/Users/Veteran/Object-recognition-using-SVM-models/'

# Feature extraction from images
for i in range(1,21,1):         # loop over particular objects' folders
    obj_nr = obj + str(i)       # storing object's number (label)
    np.random.shuffle(nr_array) # randomizing images of particular object

    # Creating two sets of earlier separated images
    training_images_ids = nr_array[:len(nr_array)//2] 
    testing_images_ids = nr_array[len(nr_array)//2:] 

    for k in range(36):         # loop over images
        # Building paths according to earlier established sets
        objpath_train = path + obj_nr + '/' + obj_nr + '__' + str(training_images_ids[k]) + '.png'
        objpath_test = path + obj_nr + '/' + obj_nr + '__' + str(testing_images_ids[k]) + '.png'

        # Loading images into separate sets (honouring labels - numbers of objects)
        training_image = cv2.imread(objpath_train)
        training_images.append([training_image, i])
        testing_image = cv2.imread(objpath_test)
        testing_images.append([testing_image, i])

    # Calculating edges for each image
    for k in range(36):
        edges_training.append([cv2.Canny(training_images[k][0],100,200), i])
        edges_testing.append([cv2.Canny(testing_images[k][0],100,200), i])

    # Calculating Hu's invariant values for each image
    for k in range(36):
        temp = np.array(cv2.HuMoments(cv2.moments(edges_training[k][0])).flatten())
        temp = np.append(temp, [i])
        Hu_training.append(temp)
        temp = np.array(cv2.HuMoments(cv2.moments(edges_testing[k][0])).flatten())
        temp = np.append(temp, [i])
        Hu_testing.append(temp)

# Classification part
Hu_training = np.array(Hu_training)
Hu_testing = np.array(Hu_testing)

X_train = Hu_training[:,:-1]  # Splitting training set into atributes and labels
y_train = Hu_training[:,-1]

X_test = Hu_testing[:,:-1]    # Splitting testing set into atributes and labels
y_test = Hu_testing[:,-1]

clf = SVC(kernel='sigmoid')                   # Initializing classifier

clf.fit(X_train, y_train)     # Training classifier and calculating the score on test set
score = accuracy_score(y_test, clf.predict(X_test))
print(score)




