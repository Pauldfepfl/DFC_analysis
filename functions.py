import numpy as np
import matplotlib.pyplot as plt
import tvb.simulator.lab as tsl
import mne_connectivity 
import tvb
from scipy import signal
from importlib import reload
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.cluster import KMeans
import scipy 
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import netplotbrain

path_conn = 'D:/Fariba/SC'
path_save = f'D:/Timing_optimisation'


# Function to compute correlation matrix for a given window
def compute_correlation_matrix(windowed_data, method= 'Pearson'):
    if method == 'Pearson': 
        correlation_matrix = np.corrcoef(windowed_data.transpose())

    if method == 'Cross-cor': 
        correlation_matrix=np.zeros((len(windowed_data),len(windowed_data)))
        cross_correlation_matrix = np.array([signal.correlate(sig, other_signal, mode='same') for sig in windowed_data for other_signal in windowed_data])
        correlation_matrix = np.corrcoef(cross_correlation_matrix)


    if method == 'Coherence': 
        f1=2.5; f2= 100
        freqs = np.linspace(f1,f2,100)
        windowed_data=windowed_data.transpose()
        windowed_data=windowed_data.reshape((1,len(windowed_data),len(windowed_data[0])))
        correlation_matrix = mne_connectivity.spectral_connectivity_time(
        windowed_data, freqs , method='coh', sfreq=1/0.0005, fmin=f1,
        fmax=f2, n_cycles=1.0, faverage=True, n_jobs=1, verbose=0).get_data(output='dense')[0,:,:,0]
    
    if method == 'Spearman':
        correlation_matrix = scipy.stats.spearmanr(windowed_data).statistic
    if method == 'Phase-lock': 
        f1=2.5; f2= 100
        freqs = np.linspace(f1,f2,100)
        windowed_data=windowed_data.transpose()
        windowed_data=windowed_data.reshape((1,len(windowed_data),len(windowed_data[0])))
        correlation_matrix = mne_connectivity.spectral_connectivity_time(
        windowed_data, freqs , method='plv', sfreq=1/0.0005, fmin=f1,
        fmax=f2, n_cycles=1.0, faverage=True, n_jobs=1, verbose=0).get_data(output='dense')[0,:,:,0]

    if method == 'wpli': 
        f1=2.5; f2= 100
        freqs = np.linspace(f1,f2,100)
        windowed_data=windowed_data.transpose()
        windowed_data=windowed_data.reshape((1,len(windowed_data),len(windowed_data[0])))
        correlation_matrix = mne_connectivity.spectral_connectivity_time(
        windowed_data, freqs , method='wpli', sfreq=1/0.0005, fmin=f1,
        fmax=f2, n_cycles=1.0, faverage=True, n_jobs=1, verbose=0).get_data(output='dense')[0,:,:,0]

    return correlation_matrix

def build_FCmat(data,method,num_clusters=3,window_size=2000,overlap=1000,tmax=30, intrahemispheric=False, square=[0,0],ts=0.5):

    # Perform windowed analysis and store correlation matrices in a list
    correlation_matrices = []
    correlation_matrix_data =[]

    if intrahemispheric : 
        for i in range(0, len(data) - int(window_size) + 1, int(window_size)-int(overlap)):
            windowed_data = data[i:i + int(window_size), :]
            correlation_matrix = compute_correlation_matrix(windowed_data,method)
            half_size=int(len(correlation_matrix)/2)
            correlation_matrices.append(np.array(correlation_matrix)[square[0]*half_size:(square[0]+1)*half_size,square[1]*half_size:(square[1]+1)*half_size])
            correlation_matrix_data.append(np.array(correlation_matrix)[square[0]*half_size:(square[0]+1)*half_size,square[1]*half_size:(square[1]+1)*half_size].flatten())
            
        # Apply K-means clustering to the correlation matrices
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        print(np.array(correlation_matrix_data).shape)
        cluster_labels = kmeans.fit_predict(correlation_matrix_data)

        # Count occurrences of each cluster label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Extract the most repeating states
        most_repeating_states = kmeans.cluster_centers_.reshape(num_clusters,half_size,half_size)
    
    else : 
        for i in range(0, len(data) - int(window_size) + 1, int(window_size)-int(overlap)):
            windowed_data = data[i:i + int(window_size), :]
            correlation_matrix = compute_correlation_matrix(windowed_data,method)
            correlation_matrices.append(correlation_matrix)
            correlation_matrix_data.append(correlation_matrix.flatten())

        # Apply K-means clustering to the correlation matrices
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        print(np.array(correlation_matrix_data).shape)
        cluster_labels = kmeans.fit_predict(correlation_matrix_data)

        # Count occurrences of each cluster label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        # Extract the most repeating states
        most_repeating_states = kmeans.cluster_centers_.reshape(num_clusters,len(correlation_matrix),len(correlation_matrix))

    # Find the closest matrix to the culster
    mat_idx = np.zeros(num_clusters)
    for j in range(len(most_repeating_states)):
        min_dist=1e100
        for i in range(len(correlation_matrices)):
            dist = np.sum(np.abs(correlation_matrices[i]-most_repeating_states[j]))
            if dist < min_dist: 
                min_dist = dist
                idx = i
        mat_idx[j] = idx 
    closest_mats=np.array(correlation_matrices)[mat_idx.astype(int)]

    # Plot the states and the closest matrices
    fig,axs=plt.subplots(1,num_clusters,figsize=(15,5))
    for a in range(len(closest_mats)):
        im=axs[a].imshow(closest_mats[a])
        axs[a].set_title(unique_labels[a])
        plt.colorbar(im,shrink=0.75)    
    fig.suptitle(f'Closest matrix to the cluster center, matrices computed with {method}, {np.round(window_size/1000*ts, 2)}s time windows', y=0.82)

    fig,axs=plt.subplots(1,num_clusters,figsize=(15,5))
    for a in range(len(most_repeating_states)):
        im=axs[a].imshow(most_repeating_states[a])
        plt.colorbar(im,shrink=0.75)
        axs[a].set_title(unique_labels[a])

    # Visualise the network if wanted 
    '''fig.suptitle(f'kmeans clusters center ', y=0.82)
    for a in range(len(closest_mats)):
        visu_network(closest_mats[a])
        visu_network(most_repeating_states[a])
    plt.show()'''
    
    return closest_mats, most_repeating_states, correlation_matrices,cluster_labels,num_clusters


# Extract the distance to each cluster
def compute_clst_dist(most_repeating_states, correlation_matrices, cluster_labels,tmax=30,num_clusters=3):
    distances = np.zeros((num_clusters,len(correlation_matrices)))
    time = np.linspace(1,tmax,len(correlation_matrices))
    for j in range((num_clusters)):
        for i in range(len(correlation_matrices)):
            distances[j,i] = np.sum(np.abs(correlation_matrices[i]-most_repeating_states[j]))

    sums=np.sum(distances,axis=0)

    for i in range(len(correlation_matrices)):
        distances[:,i] = 1-distances[:,i]/sums[i]

    colors=['red','blue','green','orange','black','purple']
    
    plt.figure(figsize=[15,10])
    
    for p in range((num_clusters)):
        plt.plot(time,distances[p,:],label=str(p),c=colors[p])
    plt.title('1-Relative distance to each cluster over time')
    plt.xlabel('Time [s]')
    plt.ylabel('1 - Relative distance')
    plt.legend()

    plt.show()

    plt.figure(figsize=[15,10])
    plt.scatter(time, cluster_labels, c = [colors[clust] for clust in cluster_labels], label=cluster_labels)
    plt.yticks(np.arange(0,num_clusters,1))
    plt.xlabel('time [s]')
    plt.ylabel('State')
    plt.title('Clustered state according to time')
    plt.show()

    
    return distances


def plot_PCA(correlation_matrices, most_repeating_states, cluster_labels,num_clusters=3):
    ext_data=np.concatenate((correlation_matrices,most_repeating_states))
    data=np.array(ext_data).reshape(len(correlation_matrices)+num_clusters,len(correlation_matrices[0])**2)
    reduced_data = PCA(n_components=2).fit_transform(data)


    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1



    u=np.diff(reduced_data[:-num_clusters, 0])
    v=np.diff(reduced_data[:-num_clusters, 1])
    pos_x = reduced_data[:-num_clusters-1, 0] + u/2
    pos_y = reduced_data[:-num_clusters-1, 1] + v/2
    norm = np.sqrt(u**2+v**2) 


    plt.plot(reduced_data[:-num_clusters, 0], reduced_data[:-num_clusters, 1],linewidth=0.5)
    sns.scatterplot(x=reduced_data[:-num_clusters, 0], y=reduced_data[:-num_clusters, 1], size=10,hue=cluster_labels,palette='Set1',legend=False)

    # Plot the centroids as a X
    centroids = reduced_data[-num_clusters:]
    sns.scatterplot(
        x=centroids[:, 0],
        y=centroids[:, 1],
        marker="x",
        s=400,
        hue=[int(x) for x in np.linspace(0,num_clusters-1,num_clusters)],
        palette='Set1'

    )

    plt.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid",width=0.001,headwidth=20)

    plt.title(
        "PCA-reduced connectivity matrices afetr K-means clustering  \n"
        "Centroids are marked with cross\n"
        f"Arrows indicate temporal order"
    )
    plt.xlim(x_min-7, x_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc='upper left')
    plt.show()

    # 3D PCA
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

    fig = go.Figure(data=[go.Scatter3d(x=reduced_data[:-num_clusters, 0], y=reduced_data[:-num_clusters, 1], z=reduced_data[:-num_clusters, 2],
                                    mode='markers', marker=dict(color=cluster_labels),ids=cluster_labels,showlegend=False)])

    centroids = reduced_data[-num_clusters:]

    fig.add_trace(go.Scatter3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                                    mode='markers+text', marker=dict(symbol='x', size=6),
                                text=[str(i) for i in range(num_clusters)], name='Centroids', showlegend=False))

    fig.add_trace(go.Scatter3d(x=reduced_data[:-num_clusters, 0],y=reduced_data[:-num_clusters, 1],z=reduced_data[:-num_clusters, 2],mode='lines',line=dict(color='black', width=2),showlegend=False))


    fig.update_layout(title=f"PCA-reduced connectivity matrices afetr K-means clustering  \n"
        "Centroids are marked with cross\n"
        f"points are linked according to temporal order", 
        height=1000, width=1000)
    fig.show()

    return pca, reduced_data 


def visu_network(connectivity_matrix,conn):
    
    bin_mat=np.zeros_like(connectivity_matrix)
    connectivity_matrix/np.max(np.abs(connectivity_matrix))
    nodes_df=pd.DataFrame(conn.centres, columns=['x','y','z'])
    below_diagonal = connectivity_matrix[np.tril_indices(connectivity_matrix.shape[0], k=-1)]
    ordered=np.sort(below_diagonal)
    bin_mat=np.where(connectivity_matrix<=np.percentile(ordered,95),0,connectivity_matrix)

    edges=[]
    for j in range(len(connectivity_matrix)) :
        for k in range(len(connectivity_matrix)) :
            if bin_mat[j][k] != 0 :
                edges.append([j,k,bin_mat[j][k]*2])
            
    edges_df=pd.DataFrame(data=edges,columns=['i','j','weight'])
    plt.figure()
    netplotbrain.plot(nodes=nodes_df,template_style = 'glass',edges=edges_df,view=['LSR'],arrowaxis=None,edge_weights=True)
    plt.show()

# Useful functions to analyse intra and interhemispheric connectivities
def square_flip(mat):
    squares=np.zeros((2,2,int(len(mat)/2),int(len(mat)/2)))
    for i in range(2) : 
        for j in range(2):
            squares[i,j]=mat[i*int(len(mat)/2):(i+1)*int(len(mat)/2),j*int(len(mat)/2):(j+1)*int(len(mat)/2)]
    mat=np.hstack([np.concatenate((squares[1,1],squares[0,1])),np.concatenate((squares[1,0],squares[0,0]))])
    return mat

def inter_flip(mat):
    squares=np.zeros((2,2,int(len(mat)/2),int(len(mat)/2)))
    for i in range(2) : 
        for j in range(2):
            squares[i,j]=mat[i*int(len(mat)/2):(i+1)*int(len(mat)/2),j*int(len(mat)/2):(j+1)*int(len(mat)/2)]
    mat=np.hstack([np.concatenate((squares[0,0],squares[0,1])),np.concatenate((squares[1,0],squares[1,1]))])
    return mat

def intra_flip(mat):
    squares=np.zeros((2,2,int(len(mat)/2),int(len(mat)/2)))
    for i in range(2) : 
        for j in range(2):
            squares[i,j]=mat[i*int(len(mat)/2):(i+1)*int(len(mat)/2),j*int(len(mat)/2):(j+1)*int(len(mat)/2)]
    mat=np.hstack([np.concatenate((squares[1,1],squares[1,0])),np.concatenate((squares[0,1],squares[0,0]))])
    return mat


def animated_plot(correlation_matrices,reduced_data,cluster_labels):
    custom_colors = ['red', 'blue', 'green']

    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Matrix Evolution', 'PCA Evolution'],specs=[
            [{"type": "Heatmap"}, {"type": "scatter3d"}]])

    # Initialize the first subplot with the initial matrix heatmap
    heatmap_trace = go.Heatmap(z=np.flip(correlation_matrices[0], axis=0), colorscale='Viridis')
    fig.add_trace(heatmap_trace, row=1, col=1)

    # Initialize the second subplot with the initial PCA scatter plot
    scatter_trace = go.Scatter3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], mode='markers',marker=dict(color=cluster_labels,size=2,colorscale=custom_colors ))
    fig.add_trace(scatter_trace, row=1, col=2)
    fig.frames=[go.Frame(data=[go.Heatmap(z=np.flip(correlation_matrices[i], axis=0), colorscale='Viridis'),go.Scatter3d(x=reduced_data[:i, 0], y=reduced_data[:i, 1], z=reduced_data[:i, 2], mode='markers',marker=dict(color=cluster_labels,size=2,colorscale=custom_colors ))], traces=[0,1]) for i in range(1,len(correlation_matrices))]
    
    # Update layout to include animation settings
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        showactive=False,
                                        buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)])
                                                    , 
                                                    {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                            "mode": "immediate",
                                                            "transition": {"duration": 0}}],
                                                            "label": "Pause",
                                                            "method": "animate"
                                                    }])],height=1000, width=2000)

    fig.show()
    pio.write_html(fig,auto_play=False,file='animated_plot.html')

def plot_PCs(pca):
    components = pca.components_
    for comp in components : 
        plt.figure()
        plt.imshow(comp.reshape(int(np.sqrt(len(comp))),int(np.sqrt(len(comp)))),cmap='bwr')
        plt.colorbar();plt.clim([-0.05,0.05])
    return components


def animated_plot_cloud(correlation_matrices,reduced_data,cluster_labels):
    custom_colors = ['red', 'blue', 'green']
    colors=[]
    sizes=[]
    for i in range(len(correlation_matrices)):
        if i<21:
            colors.append([custom_colors[cluster_labels[j]] for j in range(i+1)])
            sizes.append((i+1)*[10])
        else : 
            colors.append(['black'] * (i-20) + [custom_colors[cluster_labels[j]] for j in range(i-20,i)])
            sizes.append((i-20)*[4]+20*[10])


    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Matrix Evolution', 'PCA Evolution'],specs=[
            [{"type": "Heatmap"}, {"type": "scatter3d"}]])

    # Initialize the first subplot with the initial matrix heatmap
    heatmap_trace = go.Heatmap(z=np.flip(correlation_matrices[0], axis=0), colorscale='Viridis')
    fig.add_trace(heatmap_trace, row=1, col=1)

    # Initialize the second subplot with the initial PCA scatter plot
    scatter_trace = go.Scatter3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], mode='markers',marker=dict(color=cluster_labels,size=2,colorscale=custom_colors ))
    fig.add_trace(scatter_trace, row=1, col=2)
    fig.frames=[go.Frame(data=[go.Heatmap(z=np.flip(correlation_matrices[i], axis=0), colorscale='Viridis'),go.Scatter3d(x=reduced_data[:i, 0], y=reduced_data[:i, 1], z=reduced_data[:i, 2], mode='markers',marker=dict(color=cluster_labels,size=sizes[i],colorscale=custom_colors ))], traces=[0,1]) for i in range(1,len(correlation_matrices))]

    # Update layout to include animation settings
    fig.update_layout(updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)])
                                                    , 
                                                    {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                            "mode": "immediate",
                                                            "transition": {"duration": 0}}],
                                                            "label": "Pause",
                                                            "method": "animate"
                                                    }])],height=1000, width=2000)

    fig.show()
    pio.write_html(fig,auto_play=False,file='cloud.html')

