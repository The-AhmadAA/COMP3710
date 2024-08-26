from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch

# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# It is important in machine learning to split the data accordingly into training and testing sets to
# avoid contamination of the model. Ideally, you should also have a validation set.

# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# set up the device, load the data into tensors and send them to the device?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xtr = torch.Tensor(X_train)
xte = torch.Tensor(X_test)
ytr = torch.Tensor(y_train)
yte = torch.Tensor(y_test)

# TODO: Why do we send these to the device before they're ready?
# potential answers - operations are being performed on these Tensors
# in the device
xtr.to(device)
xte.to(device)
ytr.to(device)
yte.to(device)

# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

# Compute the PCA via eigen-decomposition of the data matrix X after the mean of the training set
# is removed. This results in a model with variations from the mean. We also transform the training and
# testing data into ’face space’, i.e. the learned sub space of the eigen-faces.

# Center data
# mean = np.mean(X_train, axis=0)
mean = torch.mean(xtr, axis=0)
xtr -= mean
xte -= mean
#Eigen-decomposition
U, S, V = torch.linalg.svd(xtr, full_matrices=False)
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))
#project into PCA subspace
X_transformed = torch.matmul(xtr, components.T)
print(X_transformed.shape)
X_test_transformed = torch.matmul(xte, components.T)
print(X_test_transformed.shape)

# Finally, plot the resulting eigen-vectors of the face PCA model, AKA the eigenfaces
import matplotlib.pyplot as plt
# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

# We should always evaluate the performance of the dimensionality reduction via a compactness plot
explained_variance = (S ** 2) / (n_samples - 1)
total_var = torch.sum(explained_variance) #.sum()
explained_variance_ratio = explained_variance / total_var

ratio_cumsum = torch.cumsum(explained_variance_ratio, 0)
print(ratio_cumsum.shape)
eigenvalueCount = torch.arange(n_components)

plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# Use the PCA ’face space’ as features and build a random forest classifier to classify the faces accord-
# ing to the labels. We then view its classification performance.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) #expects X as [n_samples, n_features]

predictions = estimator.predict(X_test_transformed)
correct = torch.from_numpy(predictions==y_test)
total_test = len(X_test_transformed)
#print("Gnd Truth:", y_test)
print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:",correct)
print("Total Correct:",torch.sum(correct))
print("Accuracy:",torch.sum(correct)/total_test)
print(classification_report(y_test, predictions, target_names=target_names, zero_division=0.0))