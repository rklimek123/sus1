import cv2 as cv
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def trim_whitespace(raw_imgs):
    cropped_images = []
    max_height = 1
    max_width = 1

    for img in raw_imgs:
        # Smallest possible Gaussian blur to get rid of noise
        blur = cv.GaussianBlur(img, (3, 3), 0)

        # Rough threshold, to get the most defining shape of the sign
        _, thresh = cv.threshold(img, 30, 255, cv.THRESH_BINARY_INV)

        # Cut the borders, leave only the most defining letter's borders
        x, y, w, h = cv.boundingRect(thresh)

        w = max(w, 1)
        h = max(h, 1)

        crop = img[y:y + h, x:x + w]

        height = crop.shape[0]
        width = crop.shape[1]

        max_width = max(max_width, width)
        max_height = max(max_height, height)

        cropped_images.append(crop)
    return cropped_images, (max_height, max_width)


def cluster(raw_imgs):
    # Crop whitespace off the images' sides.
    # Gather info about max dimensions amongst trimmed images.
    cropped_images, max_size = trim_whitespace(raw_imgs)

    matrix_rows = []

    # Unify dimensions and create a matrix of all images.
    for img in cropped_images:
        r = cv.resize(img, max_size, interpolation=cv.INTER_CUBIC)
        matrix_rows.append(r.flatten())

    X = np.asarray(matrix_rows)

    # Work out PCA
    pca = PCA(n_components=30, svd_solver="randomized")
    pca.fit(X)
    pca_X = pca.transform(X)

    # Clustering: finidng the optimal n_components
    sil_scores_kmeans = []

    begin1 = 10
    end1 = 60
    step1 = 5

    for i in range(begin1, end1 + 1, step1):
        kmeans = GaussianMixture(n_components=i, n_init=3)
        predX = kmeans.fit_predict(pca_X)
        sil_scores_kmeans.append(silhouette_score(pca_X, predX))

    ss = np.asarray(sil_scores_kmeans)
    max_ss = np.argmax(ss)
    max_ss_real = max_ss * step1 + begin1

    begin2 = max_ss_real - step1
    begin2 = max(2, begin2)
    end2 = max_ss_real + step1
    end2 = min(end1, end2)

    sil_scores_kmeans = []

    for i in range(begin2, end2 + 1):
        kmeans = KMeans(n_clusters=i)
        predX = kmeans.fit_predict(pca_X)
        sil_scores_kmeans.append(silhouette_score(pca_X, predX))

    ss = np.asarray(sil_scores_kmeans)
    max_ss = np.argmax(ss)
    max_ss_real = max_ss + begin2

    kmeans = KMeans(n_clusters=max_ss_real)
    predX = kmeans.fit_predict(pca_X)

    return predX


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 1:
        print("Usage: python3", sys.argv[0], "<list_of_images_to_cluster>")

    raw_imgs = []

    with open(sys.argv[1], "r") as f:
        for line in f:
            img = cv.imread(line.strip())
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            raw_imgs.append(img)

    predX = cluster(raw_imgs)

