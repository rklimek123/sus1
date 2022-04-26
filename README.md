# sus1
University of Warsaw, MIM Faculty, Machine Learning, Assignment 1: Character clustering

The assignment was about finding the most optimal way of clustering a dataset consisting of several thousands of characters.

## Used method

The method relies on some relatively easy concepts.

First, we trim the whitespace off the sides of each character:

1. Load image as greyscale.
2. Slightly blur the image.
3. Create a threshold mask, using absolute thresholding. (threshold on value 30) 
4. Create a bounding rectangle over the shape that came as a result of previous operations.
5. Trim the orginal image to the size and position of the bounding rectangle.
6. Measure the size of the cropped image.

Then, we have to unify the images somehow.
We look for the maximum size out of all croped images: the maximum width and the maximum height.
Then, we interpolate (using cubic interpolation, which preserves filled areas well) to the maximum size.
This way, we never shrink the image and lose data.
We only shrink if the width or height exceeds 100px.

Now, we pop these images into matrix X, each row being a vectorized matrix representing the cropped and scaled picture.

Then we can use PCA to reduce the number of dimensions to a pre-determined number. I've chosen 30.

We can begin clustering now, but we don't know the optimal target number of clusters.
We use GaussianMixture, mainly because it doesn't assume, the variances in each cluster are the same.
Therefore it can easily differentiate between the letters 'c' and 'e', something KMeans couldn't do.

We measure a silhouette score in the `range(10, 61, 5)` and pick the index `i` that returns the largest silhouette score. 
Then we look closer on the neighborhood of `i`: $[i - 5, i + 5]$. We pick the value from this neighborhood, which has the largest silhouette score.

We perform clustering with the picked value and that's our result.

## Expected time to run
Approximately 4 minutes on a good laptop.

## Custom options
No custom options.

# How to run
1. Clone the repository
2. Inside the main repository directory, run setup.sh
3. Activate the environment, `source env/bin/activate`
4. Run Python3 with the filelist containing the characters: e.g. `python3 main.py sample_filelist.txt`

The script should create two output files: `sus1.txt` and `sus1.html`.
