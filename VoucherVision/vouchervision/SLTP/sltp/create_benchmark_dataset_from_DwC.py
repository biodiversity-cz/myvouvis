import json, os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from InstructorEmbedding import INSTRUCTOR
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.cm as cm
from scipy.stats import chi2
import seaborn as sns
from collections import Counter

from SLTP_column_name_versions import ColumnVersions
from utils import validate_dir
from download_images import download_all_images_in_images_csv, bcolors

def get_cluster_name(cluster_label):
    return f"Cluster {cluster_label}"

def visualize_clusters_and_centroids(dir_plot, embeddings, cluster_labels, centroids, median_indices, extreme_indices):
    img_dim = 4000
    
    pt_scatter_2d = 5
    pt_bold_2d = 100
    pt_bold_2d_x = 50

    pt_scatter_3d = 1
    pt_scatter_3d_x = 15
    pt_bold_3d = 30



    ''' 
    # Run t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=2023)
    tsne_results = tsne.fit_transform(embeddings)

    # Cluster the data into 50 groups
    kmeans = KMeans(n_clusters=50, random_state=0).fit(tsne_results)
    cluster_labels = kmeans.labels_

    # Visualize the clusters
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=cluster_labels,
        palette=sns.color_palette("hsv", 50),
        legend="full",
        alpha=0.3
    )
    plt.title('t-SNE visualization with 50 clusters')
    plt.show()
    '''



    # Reduce the dimensionality for visualization
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(embeddings)
    reduced_centroids_3d = pca_3d.transform(centroids)

    pca_2d = PCA(n_components=2)
    reduced_embeddings_2d = pca_2d.fit_transform(embeddings)
    reduced_centroids_2d = pca_2d.transform(centroids)

    # Inspect the loadings (components)
    loadings = pca_3d.components_

    # Print the loadings for each component
    for i, component in enumerate(loadings, start=1):
        print(f"Principal Component {i}:")
        # You might want to adjust how many loadings you display
        # Here, we are showing the top 5 features for each component
        top_features = np.argsort(np.abs(component))[-5:]
        for feature in top_features:
            print(f"  Feature {feature + 1}: {component[feature]}")
        print()



    # Create a colormap based on the number of unique clusters
    unique_clusters = np.unique(cluster_labels)
    jet_colors = cm.jet(np.linspace(0, 1, unique_clusters.size))

    # Create a mapping from cluster labels to color indices
    cluster_to_color_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}

    def get_color(cluster_label):
        color_idx = cluster_to_color_idx[cluster_label]
        color = jet_colors[color_idx]
        r, g, b, a = [int(x * 255) for x in color]
        return f'rgba({r},{g},{b},{a})'


    # 3D Plot
    fig_3d = go.Figure()

    # Add scatter plot for embeddings with colors based on their cluster
    embedding_colors = [get_color(cluster_label) for cluster_label in cluster_labels]

    fig_3d.add_trace(go.Scatter3d(
        x=reduced_embeddings_3d[:, 0], 
        y=reduced_embeddings_3d[:, 1], 
        z=reduced_embeddings_3d[:, 2], 
        mode='markers',
        marker=dict(size=pt_scatter_3d, color=embedding_colors),
        name='Embeddings'
    ))

    # Plot median and extreme points for 3D
    for median_idx, extreme_idx in zip(median_indices, extreme_indices):
        median_cluster_label = cluster_labels[median_idx]
        extreme_cluster_label = cluster_labels[extreme_idx]

        median_color = get_color(median_cluster_label)
        extreme_color = get_color(extreme_cluster_label)

        median_cluster_name = get_cluster_name(median_cluster_label)
        extreme_cluster_name = get_cluster_name(extreme_cluster_label)

        # Line
        fig_3d.add_trace(go.Scatter3d(
            x=[reduced_embeddings_3d[median_idx, 0], reduced_embeddings_3d[extreme_idx, 0]],
            y=[reduced_embeddings_3d[median_idx, 1], reduced_embeddings_3d[extreme_idx, 1]],
            z=[reduced_embeddings_3d[median_idx, 2], reduced_embeddings_3d[extreme_idx, 2]],
            mode='lines',
            line=dict(color='black', width=10),
            name='Connection'
        ))
        # Median point
        fig_3d.add_trace(go.Scatter3d(
            x=[reduced_embeddings_3d[median_idx, 0]],
            y=[reduced_embeddings_3d[median_idx, 1]],
            z=[reduced_embeddings_3d[median_idx, 2]],
            mode='markers',
            marker=dict(size=pt_bold_3d, symbol='circle', color=median_color, line=dict(color='rgba(0,0,0,1)', width=2)),
            name=f'Median Point - {median_cluster_name}'
        ))
        # Extreme point
        fig_3d.add_trace(go.Scatter3d(
            x=[reduced_embeddings_3d[extreme_idx, 0]],
            y=[reduced_embeddings_3d[extreme_idx, 1]],
            z=[reduced_embeddings_3d[extreme_idx, 2]],
            mode='markers',
            # marker=dict(size=pt_bold_3d, color='rgba(0,0,0,0)', line=dict(color=extreme_color, width=100)),
            marker=dict(size=pt_scatter_3d_x, symbol='x', color=extreme_color, line=dict(color='rgba(0,0,0,1)', width=2)),
            name=f'Extreme Point - {extreme_cluster_name}'
        ))
    fig_3d.update_layout(title="3D Cluster Visualization with Centroids", scene=dict(
                        xaxis_title='PCA Component 1',
                        yaxis_title='PCA Component 2',
                        zaxis_title='PCA Component 3'))
    fig_3d.write_image(os.path.join(dir_plot, '3d_clusters_plotly.png'), width=img_dim, height=img_dim, scale=1.0)

    #### 2D Plot
    fig_2d = go.Figure()

    # Add scatter plot for embeddings with colors based on their cluster
    embedding_colors_2d = [get_color(cluster_label) for cluster_label in cluster_labels]

    fig_2d.add_trace(go.Scatter(
        x=reduced_embeddings_2d[:, 0], 
        y=reduced_embeddings_2d[:, 1], 
        mode='markers',
        marker=dict(size=pt_scatter_2d, color=embedding_colors_2d),
        name='Embeddings'
    ))

    # Plot median and extreme points for 2D
    for median_idx, extreme_idx in zip(median_indices, extreme_indices):
        median_cluster_label = cluster_labels[median_idx]
        extreme_cluster_label = cluster_labels[extreme_idx]

        median_color = get_color(median_cluster_label)
        extreme_color = get_color(extreme_cluster_label)

        median_cluster_name = get_cluster_name(median_cluster_label)
        extreme_cluster_name = get_cluster_name(extreme_cluster_label)

        # Line
        fig_2d.add_trace(go.Scatter(
            x=[reduced_embeddings_2d[median_idx, 0], reduced_embeddings_2d[extreme_idx, 0]],
            y=[reduced_embeddings_2d[median_idx, 1], reduced_embeddings_2d[extreme_idx, 1]],
            mode='lines',
            line=dict(color='black', width=4),
            name='Connection'
        ))
        # Median point
        fig_2d.add_trace(go.Scatter(
            x=[reduced_embeddings_2d[median_idx, 0]],
            y=[reduced_embeddings_2d[median_idx, 1]],
            mode='markers',
            marker=dict(size=pt_bold_2d, symbol='circle', color=median_color, line=dict(color='rgba(0,0,0,1)', width=4)),
            name=f'Median Point - {median_cluster_name}'
        ))
        # Extreme point
        fig_2d.add_trace(go.Scatter(
            x=[reduced_embeddings_2d[extreme_idx, 0]],
            y=[reduced_embeddings_2d[extreme_idx, 1]],
            mode='markers',
            marker=dict(size=pt_bold_2d_x, symbol='x', color=extreme_color, line=dict(color='rgba(0,0,0,1)', width=4)),
            name=f'Extreme Point - {extreme_cluster_name}'
        ))

    fig_2d.update_layout(title="2D Cluster Visualization with Centroids", xaxis_title='PCA Component 1', yaxis_title='PCA Component 2')
    fig_2d.write_image(os.path.join(dir_plot, '2d_clusters_plotly.png'), width=img_dim, height=img_dim, scale=1.0)


'''
def visualize_clusters_and_centroids(dir_plot, embeddings, cluster_labels, centroids, median_indices, extreme_indices):
    img_dim = 4000
    pt_scatter_2d = 25
    pt_bold_2d = 50
    pt_scatter_3d = 5
    pt_bold_3d = 30

    # Ensure the directory exists
    if not os.path.exists(dir_plot):
        os.makedirs(dir_plot, exist_ok=True)

    # Reduce the dimensionality for visualization
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(embeddings)
    reduced_centroids_3d = pca_3d.transform(centroids)

    pca_2d = PCA(n_components=2)
    reduced_embeddings_2d = pca_2d.fit_transform(embeddings)
    reduced_centroids_2d = pca_2d.transform(centroids)

    # 3D Plot
    fig_3d = go.Figure()

    # Add scatter plot for embeddings
    fig_3d.add_trace(go.Scatter3d(
        x=reduced_embeddings_3d[:, 0], 
        y=reduced_embeddings_3d[:, 1], 
        z=reduced_embeddings_3d[:, 2], 
        mode='markers',
        marker=dict(size=pt_scatter_3d, color=cluster_labels, colorscale='Jet', opacity=0.8),
        name='Embeddings'
    ))

    # Plot median and extreme points for 3D
    for median_idx, extreme_idx in zip(median_indices, extreme_indices):
        # Line
        fig_3d.add_trace(go.Scatter3d(
            x=[reduced_embeddings_3d[median_idx, 0], reduced_embeddings_3d[extreme_idx, 0]],
            y=[reduced_embeddings_3d[median_idx, 1], reduced_embeddings_3d[extreme_idx, 1]],
            z=[reduced_embeddings_3d[median_idx, 2], reduced_embeddings_3d[extreme_idx, 2]],
            mode='lines',
            line=dict(color='black', width=2),
            name='Connection'
        ))
        # Points
        fig_3d.add_trace(go.Scatter3d(
            x=[reduced_embeddings_3d[median_idx, 0], reduced_embeddings_3d[extreme_idx, 0]],
            y=[reduced_embeddings_3d[median_idx, 1], reduced_embeddings_3d[extreme_idx, 1]],
            z=[reduced_embeddings_3d[median_idx, 2], reduced_embeddings_3d[extreme_idx, 2]],
            mode='markers',
            marker=dict(size=pt_bold_3d, color=['red', 'black']),
            name='Median and Extreme Points'
        ))

    fig_3d.update_layout(title="3D Cluster Visualization with Centroids", scene=dict(
                        xaxis_title='PCA Component 1',
                        yaxis_title='PCA Component 2',
                        zaxis_title='PCA Component 3'))
    fig_3d.write_image(os.path.join(dir_plot, '3d_clusters_plotly.png'), width=img_dim, height=img_dim, scale=1.0)

    # 2D Plot
    fig_2d = go.Figure()

    # Add scatter plot for embeddings
    fig_2d.add_trace(go.Scatter(
        x=reduced_embeddings_2d[:, 0], 
        y=reduced_embeddings_2d[:, 1], 
        mode='markers',
        marker=dict(size=pt_scatter_2d, color=cluster_labels, colorscale='Jet', opacity=0.8),
        name='Embeddings'
    ))

    # Plot median and extreme points for 2D
    for median_idx, extreme_idx in zip(median_indices, extreme_indices):
        # Line
        fig_2d.add_trace(go.Scatter(
            x=[reduced_embeddings_2d[median_idx, 0], reduced_embeddings_2d[extreme_idx, 0]],
            y=[reduced_embeddings_2d[median_idx, 1], reduced_embeddings_2d[extreme_idx, 1]],
            mode='lines',
            line=dict(color='black', width=2),
            name='Connection'
        ))
        # Points
        fig_2d.add_trace(go.Scatter(
            x=[reduced_embeddings_2d[median_idx, 0], reduced_embeddings_2d[extreme_idx, 0]],
            y=[reduced_embeddings_2d[median_idx, 1], reduced_embeddings_2d[extreme_idx, 1]],
            mode='markers',
            marker=dict(size=pt_bold_2d, color=['red', 'black']),
            name='Median and Extreme Points'
        ))

    fig_2d.update_layout(title="2D Cluster Visualization with Centroids", xaxis_title='PCA Component 1', yaxis_title='PCA Component 2')
    fig_2d.write_image(os.path.join(dir_plot, '2d_clusters_plotly.png'), width=img_dim, height=img_dim, scale=1.0)
'''

def show_median_and_extreme_rows(df, embeddings):# Calculate the overall centroid of the embeddings
    try:
        successful_downloads_df = successful_downloads_df.drop(columns=['embedding'])
        df = df.drop(columns=['embedding'])
    except:
        pass
    overall_centroid = np.mean(embeddings, axis=0)

    # Calculate distances to the overall centroid
    distances_to_centroid = np.linalg.norm(embeddings - overall_centroid, axis=1)

    # Find the closest and farthest points
    closest_idx = np.argmin(distances_to_centroid)
    farthest_idx = np.argmax(distances_to_centroid)

    # Retrieve the corresponding rows from the DataFrame
    closest_row = df.iloc[closest_idx].to_dict()
    farthest_row = df.iloc[farthest_idx].to_dict()

    # Convert numpy arrays to lists for JSON serialization
    for key, value in closest_row.items():
        if isinstance(value, np.ndarray):
            closest_row[key] = value.tolist()

    for key, value in farthest_row.items():
        if isinstance(value, np.ndarray):
            farthest_row[key] = value.tolist()

    # Print as JSON objects
    print("Closest row to overall centroid:\n", json.dumps(closest_row, indent=4))
    print("\nFarthest row from overall centroid:\n", json.dumps(farthest_row, indent=4))

# Function to find median and extreme sample
def get_median_and_extreme_values(group, cluster_label, clustering_model):
    # Extract embeddings
    embeddings = np.stack(group['embedding'].values)

    # Calculate distances to the centroid
    centroid = clustering_model.cluster_centers_[cluster_label]
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Find the sample closest to the median distance
    median_distance = np.median(distances)
    median_idx = np.argmin(np.abs(distances - median_distance))
    median_sample = group.iloc[median_idx:median_idx+1]

    # Most extreme (farthest from centroid)
    extreme_idx = np.argmax(distances)
    extreme_sample = group.iloc[extreme_idx:extreme_idx+1]

    return median_sample, extreme_sample, median_idx, extreme_idx

def supplement_with_random_rows(df, sample_df, failed_clusters, sample_size):
    # Count the occurrence of each cluster in failed_clusters
    failed_clusters_filtered = [cluster for cluster in failed_clusters if cluster != ""]
    cluster_counts = Counter(failed_clusters_filtered)
    clusters_need = {i: 1 for i in range(25)}
    # Update clusters_need based on cluster_counts
    for cn, count in cluster_counts.items():
        if cn in clusters_need:
            # Add the count from cluster_counts to the value in clusters_need
            clusters_need[cn] += count

    # Extract 'id' values from sample_df
    excluded_ids = list(set(sample_df['id']))

    # Filter out rows in df that have 'id' values in excluded_ids
    df_remaining = df[~df['id'].isin(excluded_ids)]

    # # # Further filter df_remaining by rows whose 'cluster' is in failed_clusters
    # if failed_clusters:  # Check if failed_clusters is not empty
    #     df_remaining = df_remaining[df_remaining['cluster'].isin(failed_clusters)]

    # # Randomly sample sample_size rows from the remaining dataframe
    # supplement_df = df_remaining.sample(n=sample_size, random_state=2023)

    # return supplement_df, cluster_counts
    # Initialize an empty DataFrame for the supplement
    supplement_df = pd.DataFrame()

    # for cluster, count in cluster_counts.items():
    count = 1
    for cluster in range(0, 25):
        # Filter df_remaining for the current cluster
        df_cluster = df_remaining[df_remaining['cluster'] == cluster]

        # Calculate the total sample size for this cluster
        total_sample_size = sample_size * count

        # Check if the dataframe has enough rows to sample
        if len(df_cluster) >= total_sample_size:
            sampled_df = df_cluster.sample(n=total_sample_size, random_state=2023)
        else:
            # If not enough rows, take all available rows
            sampled_df = df_cluster

        # Concatenate the sampled rows to the supplement DataFrame
        supplement_df = pd.concat([supplement_df, sampled_df], ignore_index=True)

    return supplement_df, clusters_need

def download_images(df, sample_rows, sample_size, csv_kmeans_sampled):
    sample_df = pd.concat(sample_rows).reset_index(drop=True)


    successful_downloads_df, failed_clusters = download_all_images_in_images_csv(cfg_img, sample_df)


    supplement_df, clusters_need = supplement_with_random_rows(df, sample_df, failed_clusters, sample_size*10)



    if isinstance(successful_downloads_df, list):
        if all(isinstance(item, pd.DataFrame) for item in successful_downloads_df):
            successful_downloads_df = pd.concat(successful_downloads_df, ignore_index=True)
        else:
            successful_downloads_df = pd.DataFrame(successful_downloads_df)

    if isinstance(successful_downloads_df, pd.DataFrame):
        successful_downloads_df = successful_downloads_df.drop_duplicates(subset='id')
    else:
        raise ValueError("successful_downloads_df is not a DataFrame.")
    
    #### Supplement failed images with randomly selected images. Download 1 at a time until the sample_size is reached
    # supplement_iter = supplement_df.iterrows()  # Create an iterator over supplement_df

    for cluster, needed in clusters_need.items():
        while needed > 0 and successful_downloads_df.shape[0] < sample_size:
            # Create an iterator for rows of the current cluster
            supplement_iter = df[df['cluster'] == cluster]
            # supplement_iter = supplement_df[supplement_df['cluster'] == cluster]

            print(f"{bcolors.CVIOLETBG2}Supplementing with {sample_size-successful_downloads_df.shape[0]} randomly selected images{bcolors.ENDC}")
            # Get the next row from supplement_df
            try:
                next_row = supplement_iter.sample(n=1, random_state=2023)
                next_row_index = next_row.index
                # supplement_df = supplement_df.drop(next_row_index)
                df = df.drop(next_row_index)

                # Attempt to download image for the next row
                supplement_downloads_df, failed_clusters = download_all_images_in_images_csv(cfg_img, next_row.reset_index(drop=True))#pd.DataFrame([next_row]))

                if supplement_downloads_df is not None:
                    if isinstance(supplement_downloads_df, list):
                        if all(isinstance(item, pd.DataFrame) for item in supplement_downloads_df):
                            supplement_downloads_df = pd.concat(supplement_downloads_df, ignore_index=True)
                        else:
                            supplement_downloads_df = pd.DataFrame(supplement_downloads_df)

                    supplement_downloads_df = supplement_downloads_df.drop_duplicates(subset='id')
                    # If download is successful, append to successful_downloads_df
                    if not supplement_downloads_df.empty:
                        supplement_downloads_df = supplement_downloads_df.tail(1)
                        successful_downloads_df = pd.concat([successful_downloads_df, supplement_downloads_df])
                        needed -= 1
            except StopIteration:
                # Break the loop if there are no more rows to supplement
                break
        clusters_need[cluster] = needed

        if successful_downloads_df.shape[0] >= sample_size:
            break

    successful_downloads_df = successful_downloads_df.drop(columns=['embedding'])
    df = df.drop(columns=['embedding'])

    # Save the sampled dataframe to a csv file
    successful_downloads_df.to_csv(csv_kmeans_sampled, index=False)
    return df

def save_rows_as_json(dir_list, prefix='', version='MICH_to_SLTPvA'):
    path_DwC = dir_list.get('path_DwC') 
    csv_kmeans_sampled = dir_list.get('csv_kmeans_sampled') 
    csv_kmeans_sampled_SLTP = dir_list.get('csv_kmeans_sampled_SLTP') 
    dir_out = dir_list.get('dir_out') 
    dir_json_gt = dir_list.get('dir_json_gt') 
    dir_csv = dir_list.get('dir_csv') 
    dir_plot = dir_list.get('dir_plot') 
    

    CV = ColumnVersions()
    CV_conv = CV.get_conversions()
    
    # Read the dataframe from the csv file
    df = pd.read_csv(csv_kmeans_sampled, low_memory=False)

    # Filter the dataframe based on the version
    if version != 'all':
        if version == 'taxonomy':
            selected_columns = CV.get_taxonomy_columns()
        elif version == 'wcvp':
            selected_columns = CV.get_wcvp_columns()
        elif version == 'MICH_to_SLTPvA':
            selected_columns = CV.get_MICH_to_SLTPvA_columns()
        else:
            raise ValueError(f"Unknown version: {version}")

        # df = df[selected_columns]
        # Ensure all selected columns exist in df, adding them if not
        for column in selected_columns:
            if column not in df.columns:
                if column in CV_conv.keys():
                    df[column] = df[CV_conv[column]]
                else:
                    df[column] = pd.NA  # This will add the missing column with blank entries

        # Reorder df to match the order of selected_columns, adding missing ones at the end
        df = df.reindex(columns=selected_columns)

    # Fill NaN values with an empty string
    df = df.fillna('')

    # Save the filtered DataFrame as a CSV file
    df.to_csv(csv_kmeans_sampled_SLTP, index=False)
    print(f"Filtered data saved to CSV: {csv_kmeans_sampled_SLTP}")   

    # Iterate over the rows in the dataframe
    for index, row in df.iterrows():
        filename = os.path.join(dir_json_gt, prefix + str(row.iloc[0]) + '.json')
        row_dict = row.to_dict()
        with open(filename, 'w') as f:
            json.dump(row_dict, f)
        print(f"Saved JSON: {filename}")
    print("All JSON files saved.")

def create_clustered_samples(dir_list, sample_size, subset_of_images, cfg_img, model_name="hkunlp/instructor-large", visualize=True):
    path_DwC = dir_list.get('path_DwC') 
    csv_kmeans_sampled = dir_list.get('csv_kmeans_sampled') 
    dir_out = dir_list.get('dir_out') 
    dir_json_gt = dir_list.get('dir_json_gt') 
    dir_csv = dir_list.get('dir_csv') 
    dir_plot = dir_list.get('dir_plot') 
    dir_embeddings = dir_list.get('dir_embeddings')

    # Ensure n_clusters is even
    if sample_size % 2 != 0:
        sample_size += 1  # Increment to make even
        print(f"Number of clusters (sample size) adjusted to {sample_size} to ensure it is even.")

    # Load your csv data into a pandas dataframe
    df = pd.read_csv(path_DwC, low_memory=False)

    # Fill NaN values with an empty string
    df = df.fillna('')

    # Randomly sample 10% of the DataFrame using seed 2023
    df = df.sample(frac=subset_of_images, random_state=2023) 

    # Check if embeddings file exists
    fname_embeddings = '.'.join([project_name,'pkl'])
    path_embeddings = os.path.join(dir_embeddings, fname_embeddings)
    if os.path.exists(path_embeddings):
        # Load sentences & embeddings from disk
        with open(path_embeddings, "rb") as fIn:
            stored_data = pickle.load(fIn)
            sentences = stored_data['sentences']
            embeddings = stored_data['embeddings']
        print(f"Loaded embeddings from file {path_embeddings}")
    else:
        # Load model
        model = INSTRUCTOR(model_name, device="cuda")
        instruction = "Represent the Science sentence as a JSON dictionary: "

        # Generate sentences for clustering
        sentences = []
        instruction = "Represent the Science json dictionary document: "
        for row in df.itertuples():
            csv_row = ','.join(str(i) for i in row[1:])
            sentences.append([instruction, csv_row])

        # Generate embeddings
        print("Encoding rows")
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)

        # Store sentences & embeddings on disk
        with open(path_embeddings, "wb") as fOut:
            pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved embeddings to file {path_embeddings}")

    # Apply MiniBatchKMeans clustering
    sample_size_half = int(sample_size / 2)
    clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=sample_size_half, random_state=2023)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    # Add the cluster labels to the original DataFrame
    df['cluster'] = cluster_assignment

    # Add embeddings to DataFrame for processing
    df['embedding'] = list(embeddings)

    # Create a mapping from DataFrame index to embeddings index
    index_mapping = {idx: i for i, idx in enumerate(df.index)}

    # Sample median and extreme rows from each cluster
    # sample_df = df.groupby('cluster').apply(lambda x: get_median_and_extreme_values(x, clustering_model)).reset_index(drop=True)
    # Collect median and extreme indices
    median_indices = []
    extreme_indices = []
    sample_rows = []

    for cluster_label, group in df.groupby('cluster'):
        median_sample, extreme_sample, median_idx, extreme_idx = get_median_and_extreme_values(group, cluster_label, clustering_model)
        median_indices.append(index_mapping[group.index[median_idx]])
        extreme_indices.append(index_mapping[group.index[extreme_idx]])
        sample_rows.append(median_sample)
        sample_rows.append(extreme_sample)

    ################# 
    # Download images
    ################# 
    df = download_images(df, sample_rows, sample_size, csv_kmeans_sampled)

    show_median_and_extreme_rows(df, embeddings)

    if visualize:
        visualize_clusters_and_centroids(dir_plot, embeddings, cluster_assignment, clustering_model.cluster_centers_, median_indices, extreme_indices)

    ### NEED TO TRANSLATE MORTON TO DwC first
    # save_rows_as_json(dir_list, version='MICH_to_SLTPvA')
    

def morton_add_image_ID_to_original_files():
    versions = ['blue', 'green', 'wild']
    for version in versions:
        # Load the CSV files into DataFrames
        wild_df = pd.read_csv(f'D:/Dropbox/SLTP_Data/Morton/{version}_MOR_joinedDataSubsetted.csv')
        occ_df = pd.read_csv('D:/Dropbox/SLTP_Data/Morton/MOR_DwC/occurrences.csv')

        # Perform a left merge to add the 'id' from occ.csv to wild.csv based on matching 'SpecimenBarcode' and 'catalogNumber'
        merged_df = pd.merge(wild_df, occ_df[['catalogNumber', 'id']], left_on='SpecimenBarcode', right_on='catalogNumber', how='left')

        # Optionally, drop the 'catalogNumber' column if you don't need it in the final DataFrame
        merged_df.drop(columns=['catalogNumber'], inplace=True)

        # Save the modified DataFrame back to a CSV file, or proceed with further processing 

        # *** After creation I manually delete rows without ids ***
        merged_df.to_csv(f'D:/Dropbox/SLTP_Data/Morton/collection/{version}_MOR.csv', index=False)




if __name__ == '__main__':
    ### Code to merge odd formats
    # morton_add_image_ID_to_original_files()
    versions = [ 'wild'] #'blue', 'green',
    for version in versions:


        # project_dir = "D:/Dropbox/LeafMachine2/leafmachine2/transcription/benchmarks"
        # db_location = "D:/Dropbox/LeafMachine2/leafmachine2/transcription/benchmarks/chroma_DBs"
        # file_in = os.path.join(project_dir, 'candidates',f"{project_name}_CANDIDATES.csv")
        # out_dir = os.path.join(project_dir, 'selected_TEST')
        # validate_dir(out_dir)
        # validate_dir(db_location)

        subset_of_images = 1.00 #0.001 #1.0 # use this to randomly select 0-1 percent of the rows from the full occ file to speed things up (embeddings for 1 million rows takes a long time)

        sample_size = 50 # even integer
        # project_name = f'SLTP_B{sample_size}_MICH_Angiospermae2'
        project_name = f'SLTP_B{sample_size}_MOR_{version}'

        # path_DwC = 'D:/Dropbox/SLTP/datasets/MICH_Angiospermae/occurrences.csv'
        # path_DwC_multimedia = 'D:/Dropbox/SLTP/datasets/MICH_Angiospermae/multimedia.csv'
        # path_DwC_home = 'D:/Dropbox/SLTP/datasets/MICH_Angiospermae'
        # path_DwC = 'D:/Dropbox/SLTP_Data/OSU/collection/occurrences.csv'
        # path_DwC_multimedia = 'D:/Dropbox/SLTP_Data/OSU/collection/multimedia.csv'
        path_DwC = f'D:/Dropbox/SLTP_Data/Morton/collection/{version}_MOR.csv'
        path_DwC_multimedia = 'D:/Dropbox/SLTP_Data/Morton/collection/multimedia.csv'
        path_DwC_home = 'D:/Dropbox/SLTP_Data/Morton/collection'

        dir_out = 'D:/Dropbox/SLTP/benchmark_datasets'
        # dir_out = 'D:/Dropbox/SLTP/testing2'
        dir_out = os.path.join(dir_out, project_name)
        dir_json_gt = os.path.join(dir_out, 'json_gt')
        dir_csv = os.path.join(dir_out, 'benchmark_DwC')
        dir_plot = os.path.join(dir_out, 'plots')
        dir_embeddings = os.path.join(dir_out, 'embeddings')

        csv_kmeans_sampled = os.path.join(dir_csv, f"{project_name}.csv")
        csv_kmeans_sampled_SLTP = os.path.join(dir_csv, f"{project_name}_SLTPvA.csv")

        dir_list = {'path_DwC':path_DwC,
                    'csv_kmeans_sampled':csv_kmeans_sampled,
                    'csv_kmeans_sampled_SLTP':csv_kmeans_sampled_SLTP,
                    'dir_out':dir_out,
                    'dir_json_gt':dir_json_gt,
                    'dir_csv':dir_csv,
                    'dir_plot':dir_plot,
                    'dir_embeddings': dir_embeddings}
        
        ####################################################################################
        ###                         Config for image downloader                          ###
        ####################################################################################
        cfg_img = {'dir_home': dir_out,
            'path_DwC_home': path_DwC_home,
            'dir_destination_images': os.path.join(dir_out,'img'),
            'dir_destination_csv': os.path.join(dir_out,'csv'),

            ####################################################################################
            'is_custom_file': False,
            # Use for custom
            'col_url': 'url',
            'col_name': 'lab_code',
            'filename_img': 'P_serotina_79_urls.csv', # 'spines_images.txt' 'prickles_images.txt'

            ####################################################################################
            ### Exisitng Files ###
            # Darwin Core Occurance File
            'filename_occ': os.path.basename(path_DwC),

            ####################################################################################
            # Darwin Core Images File
            'filename_img': os.path.basename(path_DwC_multimedia), # 'spines_images.txt' 'prickles_images.txt'

            ####################################################################################
            ### Files that will be created ###
            # Set the filename of a new csv containing the merged records from images and occ
            #      filename_combined: 'combined_XXXXXXX.csv'
            # filename_combined: 'combined_Fagaceae.csv' # appends to this file 
            'filename_combined': 'combined_downloaded.csv',

            ####################################################################################
            ### Set bounds ###
            'MP_low': 1,
            'MP_high': 200,
            'do_resize': False,

            ####################################################################################
            ### Parallelization ###
            'n_threads': 16, # OR int in range 1:32, (usually 8-12) set to None to use number of local CPU cores + 4

            ####################################################################################
            ### Ignore Problematic Herbaria ###           download_all_images_in_images_csv(cfg) 
            # Some herbaria (MNHM, or Naturalis Biodiversity Center...) have server/broken link issues frequently
            # Set to True if you get long timeout errors
            # Recommend = True
            'ignore_banned_herb': False ,
            # You can add your own banned herbs here.
            # Add to the list based on the url stems from the 'identifier' column in the images.csv.
            # Eg. 'http://mediaphoto.mnhn.fr/media/144135118650898Vf4SNC2P5ot2SW'  -->  'mediaphoto.mnhn.fr'
            'banned_url_stems': [], #['mediaphoto.mnhn.fr'] # ['mediaphoto.mnhn.fr', 'stock.images.gov'] etc....
        }

        validate_dir([dir_out, dir_json_gt, dir_csv, dir_plot, dir_embeddings])

        create_clustered_samples(dir_list=dir_list, sample_size=sample_size, subset_of_images=subset_of_images, cfg_img=cfg_img)