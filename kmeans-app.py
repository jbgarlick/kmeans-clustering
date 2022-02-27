from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import decomposition
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris = load_iris()
iris_data = PCA(n_components=2).fit_transform(iris.data)
# %matplotlib inline

plt.rcParams["figure.dpi"] = 200

class KMeans:
    """Basic k-means clustering class."""
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-5, p=2, normalize=False):
        """Store clustering algorithm parameters.
        
        Parameters:
            n_clusters (int): How many clusters to compute.
            max_iter (int): The maximum number of iterations to compute.
            tol (float): The convergence tolerance.
            p (float): The norm to use
            normalize (bool): Whether to normalize the centers at each step
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.p = p
        self.normalize = normalize
   
    
    def fit(self, X, y=None):
        """Compute the cluster centers from random initial conditions.

        Parameters:
            X ((n_samples, n_classes) ndarray): the data to be clustered.
        """
        self.index = np.random.choice(np.arange(X.shape[0]),self.n_clusters, replace=False)
        self.cluster_centers = X[self.index,:]

        for i in range(self.max_iter):
            self.clusters = {}
            # Assign dictionary keys to be n_clusters
            for j in range(self.n_clusters):
                self.clusters[j] = []

            # Find the distance between each point and the cluster
            for rows in X:
                distance = []
                for center in self.cluster_centers:
                    distance.append(np.linalg.norm(rows-center))
                closest_center = distance.index(min(distance))
                self.clusters[closest_center].append(rows)
            
            # Store the current centers as prev_centers, so we can refer back to it after we recompute
            prev_centers = self.cluster_centers

            # Recompute the cluster centers
            for row in self.clusters:
                center_value = np.average(self.clusters[row], axis=0)
                if self.normalize:
                    norm_value = np.linalg.norm(center_value)
                    self.cluster_centers[row] = center_value/norm_value
                else:
                    self.cluster_centers[row] = center_value


            # Set checking variable to be true, will become false if we reach the tolerance level
            status = True

            # Check to see if we are within the set tolerance level
            norm_list = []
            for j in range(0,len(self.cluster_centers)):
                old_center = prev_centers[j]
                curr_center = self.cluster_centers[j]
                norm_list.append(np.linalg.norm(old_center-curr_center))
                if max(norm_list) < self.tol:
                    status = False
            if status:
                break
        return self
    
    
    def predict(self, X):
        """Classify each entry of X based on which cluster center it belongs to.

        Parameters:
            X ((n_samples, n_classes) ndarray): the data to be clustered.
        
        Returns:
            ((n_samples) ndarray): Integer labels from 0 to n_clusters-1 for each entry of X.
        """
        # Calculate the distance to each center
        distances = [np.linalg.norm(row-center) for row in self.cluster_centers]
        # Pick the closest center and label the entry of X as such
        int_label = distances.index(min(distances))
        return int_label
        
    
    def fit_predict(self, X, y=None):
        """Fit to the data and return the resulting labels.

        Parameters:
            X ((n_samples, n_classes) ndarray): the data to be clustered.
        """
        return self.fit(X).predict(X)



st.title('Unsupervised Machine Learning')
st.subheader('Using the KMeans clusting algorithm to identify patterns and groupings in data sets.')
st.write('Unsupervised machine learning is a way computers can classify data, without the need for human involvement. Typical supervised machine learning involves giving the algorithm some training data, so that it can predict what a new input is or might act like. For example, you can train an image recognition model with thousands of pictures of dogs, and then it will be able to recognize a new picture as a dog or not.')
st.write('With a model like KMeans clustering, it looks at the data and begins to learn which points are closest to each other, and figures out natural hidden groupings, sometimes referred to as clusters.')
st.write('All we need to do is input data into the model, tell it how many clusters to look for, and let it run.')
st.write('For the data plotted below, we can already see two distinct clusters of data. Select the number of clusters you want the algorithm to find, and see if it\'s what you\'d expect!')

groups = st.slider('Select the number of clusters:', 1,5,step=1)
# Create a KMeans object and fit the iris data
km = KMeans(groups)
km.fit(iris_data)

colors = 10*['r','g','c','b','k','m','y','tab:orange','tab:brown','tab:pink','tab:olive','tab:purple']
fig = plt.figure()
# Plot the centers
for center in km.cluster_centers:
    plt.scatter(center[-2], center[-1], s = 100, marker = '+', c='k')
    
# Plot the values in each cluster
for cluster in km.clusters:
    color = colors[cluster]
    for features in km.clusters[cluster]:
        plt.scatter(features[0], features[1], color = color, s = 4)
st.pyplot(fig)

st.write('The plot above has 150 points in it. Things can get a little more interesting when we are dealing with hundreds or thousands of points.')
st.write('In that case, the data aren\'t as easy to cluster with just a quick glance.')
st.write('Look at the houses in a city plotted below. Some are more structured in the city, and then some are more rural.')


data = np.load('sacramento.npy')
data = data[:,-2:]
fig3 = plt.figure()
lat = list(data[:,0])
lon = list(data[:,1])

def random_float(low, high):
    return np.random.random()*(high-low) + low

for i in range(50):
    lat.append(random_float(38.4,38.5))
    lon.append(random_float(-121.7,-121.3))

for i in range(50):
    lat.append(random_float(38.55,38.7))
    lon.append(random_float(-121.8,-121.6))

new_data = np.column_stack([lat,lon])
# plt.scatter(new_data[:,0], new_data[:,1], s=3)
# st.pyplot(fig3)

st.write('What if you were tasked, as the city planner, with placing a certain number of fire stations across the city, so that each house was as close as it could be to one of them?')
st.write('Unlike the data plotted above, doing this by hand or in your head would prove to be difficult, especially for a city with even more homes than this.')
num_groups = st.slider('Select the number of hospitals: ', min_value=1, max_value=12, step=1)
test = KMeans(n_clusters=num_groups, p=2)
test.fit_predict(new_data)
colors = 10*['r','g','c','b','k','m','y','tab:orange','tab:brown','tab:pink','tab:gray','tab:purple']

fig4 = plt.figure()
# Plot the centers
for center in test.cluster_centers:
    plt.scatter(center[-2], center[-1], s = 100, marker = '+', c='k')

# Plot the values in each cluster
for cluster in test.clusters:
    color = colors[cluster]
    for row in test.clusters[cluster]:
        plt.scatter(row[-2], row[-1], color = color, s = 4)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Sacramento')
st.pyplot(fig4)

st.write('What do you think is the optimal number to have? Building 12 might be out of the budget, but each house also needs to be close enough to a fire station so the response time is quick enough.')
st.write('Another cool application of this algorithm is finding earthquake epi-centers.')
st.write('')
st.image('earthquake_graph.png')
# earthquake_button = st.button('Run')
# if earthquake_button:
#     earthquake_data = np.load('earthquake_coordinates.npy')
#     euc_coor = []

#     for row in earthquake_data:
#         # Convert longitude and latitude to radians
#         row[0] = np.deg2rad(row[0])
#         row[1] = (np.pi/2) - np.deg2rad(row[1])
#         # Convert to spherical coordinates
#         x = np.cos(row[0])* np.sin(row[1])
#         y = np.sin(row[0])*np.sin(row[1])
#         z = np.cos(row[1])
#         euc_coor.append((x,y,z))
    
#     eq_kmeans = KMeans(15, normalize=True)
#     eq_kmeans.fit(np.array(euc_coor))


#     # Initialize the colors to be used for the plot
#     colors = 10*['r','g','c','b','k','m','y','tab:orange','tab:brown','tab:pink','tab:olive','tab:purple']

#     fig5 = plt.figure()
#     # Convert back to spherical coordinates then to degrees, plot each point
#     for cluster in eq_kmeans.clusters:
#         color = colors[cluster]
#         for row in eq_kmeans.clusters[cluster]:
#             lat = 90*np.pi - 180*np.arccos(row[2])
#             lon = (180/np.pi)*np.arctan2(row[1],row[0])
#             plt.scatter(lon, lat, color = color, s = 4)

#     # Convert each center back to spherical, then to degrees, and plot 
#     for point in eq_kmeans.cluster_centers:
#         lat = 90*np.pi - 180*np.arccos(point[2])
#         lon = (180/np.pi)*np.arctan2(point[1],point[0])
#         for center in eq_kmeans.cluster_centers:
#             plt.scatter(lon, lat, s = 100, marker = '+', c='k')
        
#     st.pyplot(fig5)

show_more = st.button('Click here to learn more about how the algorithm works.')

if show_more:
    st.write('Initially, the algorithm plots the select number of clusters. This is randomly done. So if you want 6 clusters, it picks 6 random points on the graph, and stores these values as the cluster centers.')
    st.write('Then, it goes through each point in the data set and calculates the distance from that point to each of those 6 centers, and then assigns it to the closest cluster center.')
    st.write('Each of the cluster centers is then recomputed based on the average value of the points in each cluster.')
    st.write('This is repeated again and again until the cluster centers don\'t move more than a predetermined tolerance level (usually a really small value).')


st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.text('Created by Jared Garlick')
st.write('[jaredgarlick.com](https://jaredgarlick.com)')
st.write('[LinkedIn](https://www.linkedin.com/in/jaredgarlick/)')
st.markdown('[Github](https://github.com/jbgarlick)')