import json, os, pickle
import pandas as pd
import numpy as np
import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR

"""
This script will 
- read the manually transcribed ground truth file
- embed each row using model "hkunlp/instructor-large" --- see https://huggingface.co/hkunlp/instructor-large
- use kmeans clustering to find the most dissimilar rows
- save each row as a json object, a dictionary fomatted like this:
        {
            "Genus": "Stuckenia",
            "Species": "filiformis",
            "subspecies": "",
            "variety": "",
            "forma": "",
            "Country": "Finland",
            "State": "Northern Ostrobothnia",
            "County": "Kuusamo",
            "Locality Name": "locality not transcribed for catalog no: 1177032",
            "Min Elevation": "",
            "Max Elevation": "",
            "Elevation Units": "",
            "Verbatim Coordinates": "UTM: PP2",
            "Datum": "",
            "Cultivated": "",
            "Habitat": "eutrophic fen, in a small riverlet discharging into lake Koverinj\u00e4rvi",
            "Collectors": "Marjatta Aalto",
            "Collector Number": "3516",
            "Verbatim Date": "8 August 1976",
            "Date": "1976-08-08",
            "End Date": ""
        }
- these .json files can then be used with the evaluate_LLM_predictions.py methods as a benchmark

"""
def validate_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

# Function to find median and extreme sample
def find_samples(group, clustering_model):
    # Median
    median_idx = group['embedding'].median()
    median_sample = group.iloc[(group['embedding'] - median_idx).abs().argsort()[:1]]

    # Most extreme (farthest from centroid)
    centroid = clustering_model.cluster_centers_[group.name]
    distances = np.linalg.norm(group['embedding'].to_list() - centroid, axis=1)
    extreme_sample = group.iloc[np.argmax(distances)]

    return pd.concat([median_sample, extreme_sample])

def create_clustered_samples(project_name, input_csv, output_csv, n_clusters, dir_embeddings, model_name="hkunlp/instructor-large"):
    validate_dir(dir_embeddings)
    # Ensure n_clusters is even
    if n_clusters % 2 != 0:
        n_clusters += 1  # Increment to make even
        print(f"Number of clusters (sample size) adjusted to {n_clusters} to ensure it is even.")

    # Load model
    model = INSTRUCTOR(model_name, device="cuda") # TODO: ensure the correct device is used

    # Load CSV data into a pandas DataFrame
    df = pd.read_csv(input_csv)
    df = df.fillna('')  # Fill NaN values

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
        # Generate sentences for clustering
        sentences = []
        instruction = "Represent the Science json dictionary document: "
        for row in df.itertuples():
            csv_row = ','.join(str(i) for i in row[1:])
            sentences.append([instruction, csv_row])

        # Generate embeddings
        print("Encoding rows")
        embeddings = model.encode(sentences, batch_size=48, show_progress_bar=True)

        # Store sentences & embeddings on disk
        with open(path_embeddings, "wb") as fOut:
            pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved embeddings to file {path_embeddings}")

    # Apply MiniBatchKMeans clustering
    n_clusters_half = int(n_clusters / 2)
    clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters_half, random_state=2023)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    # Add the cluster labels to the original DataFrame
    df['cluster'] = cluster_assignment

    # Add embeddings to DataFrame for processing
    df['embedding'] = list(embeddings)

    # Sample median and extreme rows from each cluster
    sample_df = df.groupby('cluster').apply(find_samples).reset_index(drop=True)

    # Save the sampled dataframe to a csv file
    sample_df.to_csv(output_csv, index=False)

'''
Will create a dir that contains 
'''
def save_rows_as_json(input_csv, prefix='', dir_out=None):
    # Read the dataframe from the csv file
    df = pd.read_csv(input_csv)

    # Fill NaN values with an empty string
    df = df.fillna('')

    if dir_out is None:
        # Determine the name of the new directory
        parent_dir = os.path.dirname(input_csv)
        filestem = os.path.splitext(os.path.basename(input_csv))[0]
        new_dir = os.path.join(parent_dir, filestem)
    else:
        new_dir = dir_out

    # Create the new directory
    os.makedirs(new_dir, exist_ok=True)

    # Create an empty list to store the dictionaries
    # dict_list = []

    # Iterate over the rows in the dataframe
    for index, row in df.iterrows():
        # Use the value in the first column as the file name
        # filename = os.path.join(new_dir, prefix + str(row.iloc[0]) + '.json')  # **** TODO do all columns for GBIF style...?
        filename = os.path.join(new_dir, prefix + str(row.iloc[0]) + '.json')  # **** TODO do all columns for GBIF style...?

        # Convert the row (excluding the first and last column) to a dictionary
        row_dict = row.iloc[0:].to_dict() # **** TODO do all columns for GBIF style...?
        # row_dict = row.to_dict()

        # Append the dictionary to the list
        # dict_list.append(row_dict)

        # Write the dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(row_dict, f)

        # # Use a prefix for the combined json filename
        # filename = os.path.join(new_dir, prefix + 'combined' + '.json')

        # # Write the list of dictionaries to a single JSON file
        # with open(filename, 'w') as f:
        #     json.dump(dict_list, f)

def create_JSON_groundtruth():
    # prefix = 'MICH-V-'
    prefix = ''

    # path_in = 'D:\Dropbox\LM2_Env\VoucherVision_Datasets\Set2_Easier_MI/2022_02_01_S3_MI_mklemz__TRIMMED.csv'
    # dir_out = 'D:\Dropbox\LM2_Env\VoucherVision_Datasets\GT__Set2_Easier_MI'

    path_in = 'D:\Dropbox\LM2_Env\VoucherVision_Datasets\Set1_Difficult_AllAsia/2022_10_26_S3_jacortez_AllAsiaOnly_Cyper_TRIMMED.csv'
    dir_out = 'D:\Dropbox\LM2_Env\VoucherVision_Datasets\GT__Set1_Difficult_AllAsia'

    save_rows_as_json(path_in, prefix, dir_out)

def create_benchmark_dataset():
    ### Creating a benchmark  dataset
    # file_in = 'D:/Dropbox/LeafMachine2/leafmachine2/transcription/domain_knowledge/AllAsiaMinimalasof25May2023_2__InRegion.csv'
    # out_dir = 'D:/Dropbox/LeafMachine2/leafmachine2/transcription/domain_knowledge/'

    # file_out = f'{out_dir}/HLT_C21_AA_S10.csv'
    # create_clustered_samples(file_in, file_out, 10)
    # save_rows_as_json(file_out, 'HLT_C21_AA_S10', os.path.join(out_dir, 'HLT_C21_AA_S10'))
    
    # file_out = f'{out_dir}/HLT_C21_AA_S50.csv'
    # create_clustered_samples(file_in, file_out, 50)
    # save_rows_as_json(file_out, 'HLT_C21_AA_S50', os.path.join(out_dir, 'HLT_C21_AA_S50'))

    # file_out = f'{out_dir}/HLT_C21_AA_S200.csv'
    # create_clustered_samples(file_in, file_out, 200)
    # save_rows_as_json(file_out, 'HLT_C21_AA_S200', os.path.join(out_dir, 'HLT_C21_AA_S200'))

    project_dir = "D:/Dropbox/LeafMachine2/leafmachine2/transcription/benchmarks/"
    sample_size = 75

    # project_name = 'SLTP_B50_MICH_Indonesia'
    # file_in = os.path.join(project_dir, 'candidates',f"{project_name}_CANDIDATES.csv")
    # out_dir = os.path.join(project_dir, 'selected')

    # file_out = os.path.join(out_dir, f"{project_name}.csv")
    # create_clustered_samples(file_in, file_out, sample_size)
    # save_rows_as_json(file_out, project_name, os.path.join(out_dir, project_name))

    # project_name = 'SLTP_B50_MICH_Malaysia'
    # file_in = os.path.join(project_dir, 'candidates',f"{project_name}_CANDIDATES.csv")
    # out_dir = os.path.join(project_dir, 'selected')

    # file_out = os.path.join(out_dir, f"{project_name}.csv")
    # create_clustered_samples(file_in, file_out, sample_size)
    # save_rows_as_json(file_out, project_name, os.path.join(out_dir, project_name))

    project_name = 'SLTP_B50_MICH_MI'
    file_in = os.path.join(project_dir, 'candidates',f"{project_name}_CANDIDATES.csv")
    out_dir = os.path.join(project_dir, 'selected')

    file_out = os.path.join(out_dir, f"{project_name}_TESTTT.csv")
    dir_embeddings = out_dir = os.path.join(project_dir, 'selected', 'embeddings')
    create_clustered_samples(project_name, file_in, file_out, sample_size, dir_embeddings)
    save_rows_as_json(file_out, project_name, os.path.join(out_dir, project_name))

if __name__ == '__main__':

    create_benchmark_dataset()

    ### !!!!!
    ### Make sure that Excel has not misformatted dates!
    ### Usually opening the xlsx, saving a copy as a csv, removing/editing the column names is good
    # create_JSON_groundtruth()