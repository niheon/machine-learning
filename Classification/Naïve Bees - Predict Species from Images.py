#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Python libraries
# <p><img src="https://assets.datacamp.com/production/project_412/img/92_notebook.jpg" alt="honey bee">
# <em>A honey bee (Apis).</em></p>
# <p>Can a machine identify a bee as a honey bee or a bumble bee? These bees have different <a href="https://www.thesca.org/connect/blog/bumblebees-vs-honeybees-what%E2%80%99s-difference-and-why-does-it-matter">behaviors and appearances</a>, but given the variety of backgrounds, positions, and image resolutions, it can be a challenge for machines to tell them apart.</p>
# <p>Being able to identify bee species from images is a task that ultimately would allow researchers to more quickly and effectively collect field data. Pollinating bees have critical roles in both ecology and agriculture, and diseases like <a href="http://news.harvard.edu/gazette/story/2015/07/pesticide-found-in-70-percent-of-massachusetts-honey-samples/">colony collapse disorder</a> threaten these species. Identifying different species of bees in the wild means that we can better understand the prevalence and growth of these important insects.</p>
# <p><img src="https://assets.datacamp.com/production/project_412/img/20_notebook.jpg" alt="bumble bee">
# <em>A bumble bee (Bombus).</em></p>
# <p>After loading and pre-processing images, this notebook walks through building a model that can automatically detect honey bees and bumble bees.</p>

# In[75]:


# used to change filepaths
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

# import Image from PIL
from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import train_test_split from sklearn's model selection module
from sklearn.model_selection import train_test_split

# import SVC from sklearn's svm module
from sklearn.svm import SVC

# import accuracy_score from sklearn's metrics module
from sklearn.metrics import roc_curve, auc, accuracy_score


# In[76]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task_1():\n    assert 'Image' in globals(), \\\n    'Did you forget to import `Image` from `PIL`?'\n    \ndef test_task_2():\n    assert 'train_test_split' in globals(), \\\n    'Did you forget to import `train_test_split` from `sklearn.model_selection`?'\n    \ndef test_task_3():\n    assert 'SVC' in globals(), \\\n    'Did you forget to import `SVC` from `sklearn.svm`?'\n    \ndef test_task_4():\n    assert 'accuracy_score' in globals(), \\\n    'Did you forget to import `accuracy_score` from `sklearn.metrics`?'")


# ## 2. Display image of each bee type
# <p>Now that we have all of our imports ready, it is time to look at some images. We will load our <code>labels.csv</code> file into a dataframe called <code>labels</code>, where the index is the image name (e.g. an index of 1036 refers to an image named 1036.jpg) and the <code>genus</code> column tells us the bee type. <code>genus</code> takes the value of either <code>0.0</code> (Apis or honey bee) or <code>1.0</code> (Bombus or bumble bee).</p>
# <p>The function <code>get_image</code> converts an index value from the dataframe into a file path where the image is located, opens the image using the <a href="https://pillow.readthedocs.io/en/5.1.x/reference/Image.html">Image</a> object in Pillow, and then returns the image as a numpy array.</p>
# <p>We'll use this function to load the sixth Apis image and then the sixth Bombus image in the dataframe.</p>

# In[77]:


# load the labels using pandas
labels = pd.read_csv("datasets/labels.csv", index_col=0)

# show the first five rows of the dataframe using head
display(labels.head())

def get_image(row_id, root="datasets/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)

# subset the dataframe to just Apis (genus is 0.0) get the value of the sixth item in the index
apis_row = labels[labels.genus == 0.0].index[5]

# show the corresponding image of an Apis
plt.imshow(get_image(apis_row))
plt.show()

# subset the dataframe to just Bombus (genus is 1.0) get the value of the sixth item in the index
bombus_row = labels[labels.genus == 1.0].index[5]

# show the corresponding image of a Bombus
plt.imshow(get_image(bombus_row))
plt.show()


# In[78]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task2_0():\n    assert (bombus_row == 1934), \\\n    'Did you get the sixth row of the index of the subsetted dataframe (labels[labels.genus == 0.0])?'")


# ## 3. Image manipulation with rgb2gray
# <p>scikit-image has a number of image processing functions built into the library, for example, converting an image to grayscale. The <a href="https://scikit-image.org/docs/dev/api/skimage.color.html?highlight=rgb2#skimage.color.rgb2gray"><code>rgb2gray</code></a> function computes the luminance of an RGB image using the following formula <code>Y = 0.2125 R + 0.7154 G + 0.0721 B</code>. </p>
# <p>Image data is represented as a matrix, where the depth is the number of channels. An RGB image has three channels (red, green, and blue) whereas the returned grayscale image has only one channel. Accordingly, the original color image has the dimensions <code>100x100x3</code> but after calling <code>rgb2gray</code>, the resulting grayscale image has only one channel, making the dimensions <code>100x100x1</code>.</p>

# In[79]:


# load a bombus image using our get_image function and bombus_row from the previous cell
bombus = get_image(bombus_row)

# print the shape of the bombus image
print('Color bombus image has shape: ', bombus.shape)

# convert the bombus image to grayscale
gray_bombus = rgb2gray(bombus)

# show the grayscale image
plt.imshow(gray_bombus, cmap=mpl.cm.gray)

# confirm grayscale bombus image only has one channel
print('Grayscale bombus image has shape: ', gray_bombus.shape)


# In[80]:


get_ipython().run_cell_magic('nose', '', "import numpy\n\ndef test_task3_0():\n    assert 'bombus' in globals() and bombus.shape == (100, 100, 3), \\\n    'Did you load the image corresponding to `bombus_row` using the `get_image` function and assign it to `bombus`?'\n    \ndef test_task3_1():\n    assert gray_bombus.shape == (100, 100) and gray_bombus.max() <= 1, \\\n    'Did you convert `bombus` to grayscale using `rgb2gray`?'")


# ## 4. Histogram of oriented gradients
# <p>Now we need to turn these images into something that a machine learning algorithm can understand. Traditional computer vision techniques have relied on mathematical transforms to turn images into useful features. For example, you may want to detect edges of objects in an image, increase the contrast, or filter out particular colors.</p>
# <p>We've got a matrix of pixel values, but those don't contain enough interesting information on their own for most algorithms. We need to help the algorithms along by picking out some of the salient features for them using the <a href="http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html">histogram of oriented gradients</a> (HOG) descriptor. The idea behind <a href="https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients">HOG</a> is that an object's shape within an image can be inferred by its edges, and a way to identify edges is by looking at the direction of intensity gradients (i.e. changes in luminescence). </p>
# <p><img src="https://assets.datacamp.com/production/project_412/img/hog.png" alt="HOG"></p>
# <p>An image is divided in a grid fashion into cells, and for the pixels within each cell, a histogram of gradient directions is compiled. To improve invariance to highlights and shadows in an image, cells are block normalized, meaning an intensity value is calculated for a larger region of an image called a block and used to contrast normalize all cell-level histograms within each block. The HOG feature vector for the image is the concatenation of these cell-level histograms.</p>

# In[81]:


# run HOG using our grayscale bombus image
hog_features, hog_image = hog(gray_bombus,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

# show our hog_image with a gray colormap
plt.imshow(hog_image, cmap=mpl.cm.gray)


# In[82]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task4_0():\n    assert all(hog_image[0] == np.array([0] * 100)), \\\n    'Did you call `hog` on `gray_bombus`?'\n    \ndef test_task4_1():\n    assert '_' in globals() and isinstance(globals()['_'], mpl.image.AxesImage), \\\n    'Did you forget to call `plt.imshow` on `hog_image`?'")


# ## 5. Create image features and flatten into a single row
# <p>Algorithms require data to be in a format where rows correspond to images and columns correspond to features. This means that all the information for a given image needs to be contained in a single row.</p>
# <p>We want to provide our model with the raw pixel values from our images as well as the HOG features we just calculated. To do this, we will write a function called <code>create_features</code> that combines these two sets of features by flattening the three-dimensional array into a one-dimensional (flat) array.</p>

# In[83]:


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to grayscale
    gray_image = rgb2gray(img)
    # get HOG features from grayscale image
    hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack([color_features, hog_features])
    return flat_features

bombus_features = create_features(bombus)

# print shape of bombus_features
bombus_features.shape


# In[84]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task5_0():\n    assert bombus_features[29999] == 118.0 and round(bombus_features[30000], 3) == 0.053, \\\n    'Did you pass color_features and hog_featuers into the np.hstack function in this order?'\n    \ndef test_task5_1():\n    assert bombus_features.shape == (31296,), \\\n    '`bombus_features` does not have the correct shape. Did you setup the `create_features` function properly?'")


# ## 6. Loop over images to preprocess
# <p>Above we generated a flattened features array for the bombus image. Now it's time to loop over all of our images. We will create features for each image and then stack the flattened features arrays into a big matrix we can pass into our model.</p>
# <p>In the <code>create_feature_matrix</code> function, we'll do the following:</p>
# <ul>
# <li>Load an image</li>
# <li>Generate a row of features using the <code>create_features</code> function above</li>
# <li>Stack the rows into a features matrix</li>
# </ul>
# <p>In the resulting features matrix, rows correspond to images and columns to features.</p>

# In[85]:


def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        # get features
        image_features = create_features(img)
        features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(labels)


# In[86]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task6_0():\n    assert feature_matrix[0, -1] != feature_matrix[1, -1], \\\n    'Did you call `create_features` on `img`?'\n    \ndef test_task6_1():\n    assert feature_matrix.shape == (500, 31296), \\\n    'Did you call `create_feature_matrix` on the dataframe `labels`?'")


# ## 7. Split into train and test sets
# <p>Now we need to convert our data into train and test sets. We'll use 70% of images as our training data and test our model on the remaining 30%. Scikit-learn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"><code>train_test_split</code></a> function makes this easy.</p>

# In[87]:


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix,
                                                    labels.genus.values,
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distribution of labels in the train set
pd.Series(y_train).value_counts()


# In[88]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task7_0():\n    assert '_' in globals() and isinstance(globals()['_'], pd.Series), \\\n    'Did you forget to forget to look at the distribution of labels using `pd.Series(y_train).value_counts()`?'\n    \ndef test_task7_1():\n    assert X_train.shape == (350, 31296), \\\n    'Did you pass `feature_matrix` as X into train_test_split?'\n                                     \ndef test_task7_2():\n    assert y_train.shape == (350,) and (np.unique(y_train) == [0., 1.,]).all(), \\\n    'Did you pass `labels.genus.values` as y into train_test_split?'")


# ## 8. Scale training features + PCA
# <p>Our features aren't quite done yet. Many machine learning methods are built to work best with data that has a mean of 0 and unit variance. Luckily, scikit-learn <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">provides a simple way</a> to rescale your data to work well using <code>StandardScaler</code>. They've got a more thorough explanation of why that is in the linked docs.</p>
# <p>Remember also that we have over 31,000 features for each image and only 500 images total. To use an SVM, our model of choice, we also need to reduce the number of features we have using <a href="http://scikit-learn.org/stable/modules/decomposition.html#pca">principal component analysis</a> (PCA). </p>
# <p>PCA is a way of linearly transforming the data such that most of the information in the data is contained within a smaller number of features called components. Below is a visual <a href="https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60">example</a> from an image dataset containing handwritten numbers. The image on the left is the original image with 784 components. We can see that the image on the right (post PCA) captures the shape of the number quite effectively even with only 59 components.</p>
# <p><img src="https://assets.datacamp.com/production/project_412/img/pca_cropped.png" alt="PCA"></p>
# <p>In our case, we will keep 350 components. This means our feature matrices <code>X_train</code> and <code>y_train</code> will only have 350 columns, rather than the original of 31,296. Let's start by scaling and performing PCA on our training features. </p>

# In[89]:


# get shape of our training features
print('Training features matrix shape is: ', X_train.shape)

# define standard scaler
ss = StandardScaler()

# fit the scaler and transform the training features
train_stand = ss.fit_transform(X_train)

pca = PCA(n_components=350)

# use fit_transform to run PCA on our standardized training features
X_train = pca.fit_transform(train_stand)

# look at new shape
print('Training features matrix shape is: ', X_train.shape)


# In[90]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task8_0():\n    assert round(train_stand[0, 0], 3) == -1.124, \\\n    'Did you pass in `X_train` to `ss_train.fit_transform()`?'\n    \ndef test_task8_1():\n    assert round(X_train[0, 0], 3) == -45.062, \\\n    'Did you pass in `train_stand` to `pca_train.fit_transform()`?'\n    \ndef test_task8_2():\n    assert X_train.shape == (350, 350), \\\n    'Did you pass in `train_stand` to `pca_train.fit_transform()`?'")


# ## 9. Scale test features + PCA
# <p>Now we have scaled and performed PCA on our training features, we need to must do the same with the test set to ensure the same scale is applied to both sets, and the same volume of features used. </p>
# <p>We will use the <code>ss</code> and <code>pca</code> variables fit to the training features, to transform the test features.</p>

# In[91]:


# get shape of our test data
print('Test features matrix shape is: ', X_test.shape)

# use transform our test features
test_stand = ss.transform(X_test)

# use transform on our standardized test features
X_test = pca.transform(test_stand)

# look at new shape
print('Test features matrix shape is: ', X_test.shape)


# In[92]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task9_0():\n    assert round(test_stand[0, 0], 3) == -0.749, \\\n    'Did you pass in `X_test` to `ss_test.fit_transform()`?'\n    \ndef test_task9_1():\n    assert round(X_test[0, 0], 3) == 29.419, \\\n    'Did you pass in `test_stand` to `pca_test.fit_transform()`?'\n    \ndef test_task9_2():\n    assert X_test.shape == (150, 350), \\\n    'Did you pass in `test_stand` to `pca_test.fit_transform()`?'")


# ## 10.Train model
# <p>It's finally time to build our model! We'll use a <a href="http://scikit-learn.org/stable/modules/svm.html">support vector machine</a> (SVM), a type of supervised machine learning model used for regression, classification, and outlier detection." An <a href="https://en.wikipedia.org/wiki/Support_vector_machine">SVM model</a> is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall."</p>
# <p>Here's a visualization of the maximum margin separating two classes using an SVM classifier with a linear kernel.
# <img src="https://assets.datacamp.com/production/project_412/img/svm_cropped.png" alt="SVM"></p>
# <p>Since we have a classification task -- honey or bumble bee -- we will use the support vector classifier (SVC), a type of SVM. We imported this class at the top of the notebook.</p>

# In[93]:


# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)


# In[94]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task10_0():\n    assert svm.kernel == 'linear' and svm.probability == True, \\\n    'Did you assign define an SVC with a linear kernel and set probability equal to True?'")


# ## 11. Score model
# <p>Now we'll use our trained model to generate predictions for our test data. To see how well our model did, we'll calculate the accuracy by comparing our predicted labels for the test set with the true labels in the test set. Accuracy is the number of correct predictions divided by the total number of predictions. Scikit-learn's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"><code>accuracy_score</code></a> function will do math for us. Sometimes accuracy can be misleading, but since we have an equal number of honey and bumble bees, it is a useful metric for this problem.</p>

# In[95]:


# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)


# In[96]:


get_ipython().run_cell_magic('nose', '', "    \ndef test_task11_1():\n    assert pd.Series(y_pred).value_counts()[0] == 79, \\\n    'Did you generate predictions using `svm.predict(X_test)`?'\n    \ndef test_task11_2():\n    assert round(accuracy, 2) == 0.68, \\\n    'Did you calculate accuracy using `accuracy_score(y_test, y_pred)`?'")


# ## 12. ROC curve + AUC
# <p>Above, we used <code>svm.predict</code> to predict either <code>0.0</code> or <code>1.0</code> for each image in <code>X_test</code>. Now, we'll use <code>svm.predict_proba</code> to get the probability that <strong>each class</strong> is the true label. For example, <code>predict_proba</code> returns <code>[0.46195176, 0.53804824]</code> for the first image, meaning there is a 46% chance the bee in the image is an Apis (<code>0.0</code>) and a 53% chance the bee in the image is a Bombus (<code>1.0</code>). Note that the two probabilities for each image always sum to 1. </p>
# <p>Using the default settings, probabilities of 0.5 or above are assigned a class label of <code>1.0</code> and those below are assigned a <code>0.0</code>. However, this threshold can be adjusted. The <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic">receiver operating characteristic curve</a> (ROC curve) plots the false positive rate and true positive rate at different thresholds. ROC curves are judged visually by how close they are to the upper lefthand corner. </p>
# <p>The <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve">area under the curve</a> (AUC) is also calculated, where 1 means every predicted label was correct. Generally, the worst score for AUC is 0.5, which is the performance of a model that randomly guesses. See the scikit-learn documentation for more resources and examples on <a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html">ROC curves</a> and <a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html">AUC</a>.</p>

# In[97]:


# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');


# In[98]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task12_0():\n    assert round(y_proba[0], 1) == .5 , \\\n    'Did you define probabilities using `svm.predict_proba(X_test) and y_proba using `probabilityes[:, 1]?'\n\ndef test_task12_1():\n    assert false_positive_rate.shape == (74,), \\\n    'Did you pass `y_test, y_proba, pos_label=1` into roc_curve?'\n    \ndef test_task12_2():\n    assert round(roc_auc, 2) == .74, \\\n    'Did you calculate the roc_auc properly?'\n    \ndef test_task12_3():\n    x, y = roc_plot[0].get_data()\n    assert x[10] == 0.08 and round(y[10], 2) == 0.28, \\\n    'Did you plot the false positive rate on the x axis and the true positive rate on the y axis?'")

