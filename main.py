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
    return cropped_images, (min(100, max_height), min(100, max_width))


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

    begin1 = 20
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


def create_txt_report(filenames, pred):
    n = pred.size
    ids = np.arange(0, n)
    report = ""

    for cluster in np.unique(pred):
        fnames = np.asarray(filenames)[ids[pred == cluster]]
        fline = " ".join(fnames) + "\n"
        report += fline

    with open("sus1.txt", "w") as f:
        f.write(report)


def create_html_report(img_paths, pred):
    n = pred.size
    ids = np.arange(0, n)
    report = "<html><head><title>Characters clustering</title></head><body>"

    for cluster in np.unique(pred):
        fnames = np.asarray(img_paths)[ids[pred == cluster]]

        report += "<div>"

        for fname in fnames:
            report += "<img src=\"" + fname + "\"/>"

        report += "</div><hr/>\n"

    report += "</body></html>"

    with open("sus1.html", "w") as f:
        f.write(report)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 2:
        print("Usage: python3", sys.argv[0], "<list_of_images_to_cluster>")
        exit(1)

    raw_imgs = []
    filenames = []
    fullfilenames = []

    with open(sys.argv[1], "r") as f:
        for line in f:
            l = line.strip()
            img = cv.imread(l)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            raw_imgs.append(img)

            _, fname = os.path.split(l)
            filenames.append(fname)
            fullfilenames.append(l)

    predX = cluster(raw_imgs)
    create_txt_report(filenames, predX)
    create_html_report(fullfilenames, predX)

