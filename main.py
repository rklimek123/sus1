import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

directory = 'probka_uczaca_zad2'

raw_imgs = []
cropped_images = []

max_width = 0
max_height = 0

# Crop whitespace off the images' sides.
# Gather info about max dimensions.
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    img = cv.imread(f)

    # Convert to grayscale from default BGR
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    raw_imgs.append(img)

    # Smallest possible Gaussian blur to get rid of noise
    blur = cv.GaussianBlur(img, (3, 3), 0)

    # Rough threshold, to get the most defining shape of the sign
    _, thresh = cv.threshold(img, 30, 255, cv.THRESH_BINARY_INV)

    # Cut the borders, leave only the most defining letter's borders
    x, y, w, h = cv.boundingRect(thresh)

    w = max(w, 1)
    h = max(h, 1)

    crop = img[y:y+h, x:x+w]

    height = crop.shape[0]
    width = crop.shape[1]

    max_width = max(max_width, width)
    max_height = max(max_height, height)

    cropped_images.append(crop)

matrix_rows = []

# Unify dimensions and create a matrix of all images.
for img in cropped_images:
    r = cv.resize(img, (max_width, max_height), interpolation=cv.INTER_CUBIC)
    matrix_rows.append(r.flatten())

X = np.asarray(matrix_rows)

# Work out PCA
pca = PCA(n_components=30, svd_solver="randomized")
pca.fit(X)
pca_X = pca.transform(X)


def show_matrix(mat, mat2=None, dirty=False, label_data=None, no_reformat_second=False):
    if mat2 is None:
        for i in range(mat.shape[0]):
            row = mat[i]
            if label_data is not None:
                print("Label:",label_data[i])

            img = row.copy()
            img.resize((max_height, max_width))
            if dirty:
                plt.imshow(img)
            else:
                plt.imshow(img, 'gray', vmin=0, vmax=255)
            plt.show()
    else:
        for i in range(mat.shape[0]):
            if label_data is not None:
                print("Label:",label_data[i])

            img1 = mat[i].copy()
            img1.resize((max_height, max_width))

            if not no_reformat_second:
                img2 = mat2[i].copy()
                img2.resize((max_height, max_width))
            else:
                img2 = mat2[i]

            plt.subplot(2, 1, 1)
            if dirty:
                plt.imshow(img1)
            else:
                plt.imshow(img1, 'gray', vmin=0, vmax=255)
            plt.subplot(2, 1, 2)
            if dirty:
                plt.imshow(img2)
            else:
                plt.imshow(img2, 'gray', vmin=0, vmax=255)
            plt.show()

#show_matrix(pca.components_, dirty=True)
#show_matrix(X, pca.inverse_transform(pca_X))

# Clustering
sil_scores_kmeans = []

begin1 = 20
end1 = 50
step1 = 5

for i in range(begin1, end1 + 1, step1):
    print("Cluster", i)
    kmeans = GaussianMixture(n_components=i, n_init=3)
    predX = kmeans.fit_predict(pca_X)
    sil_scores_kmeans.append(silhouette_score(pca_X, predX))

plt.plot(np.arange(begin1, end1 + 1, step1), sil_scores_kmeans)
plt.show()

ss = np.asarray(sil_scores_kmeans)
max_ss = np.argmax(ss)
max_ss_real = max_ss * step1 + begin1

begin2 = max_ss_real - step1
begin2 = max(2, begin2)
end2 = max_ss_real + step1
end2 = min(end1, end2)

sil_scores_kmeans = []

for i in range(begin2, end2 + 1):
    print("Cluster", i)
    kmeans = KMeans(n_clusters=i)
    predX = kmeans.fit_predict(pca_X)
    sil_scores_kmeans.append(silhouette_score(pca_X, predX))

plt.plot(np.arange(begin2, end2 + 1), sil_scores_kmeans)
plt.show()

ss = np.asarray(sil_scores_kmeans)
max_ss = np.argmax(ss)
max_ss_real = max_ss + begin2

kmeans = KMeans(n_clusters=max_ss_real)
predX = kmeans.fit_predict(pca_X)

#show_matrix(pca.inverse_transform(pca_X), raw_imgs, label_data=predX, no_reformat_second=True)

for i in range(max_ss_real):
    p1 = pca.inverse_transform(pca_X)[predX == i]
    p2 = np.asarray(raw_imgs)[predX == i]
    try:
        show_matrix(p1, p2, label_data=predX[predX == i], no_reformat_second=True)
    except KeyboardInterrupt:
        continue
