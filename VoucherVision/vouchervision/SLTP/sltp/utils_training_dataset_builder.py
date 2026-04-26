import os, random, json, yaml
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from huggingface_hub import HfApi, Repository
from datasets import load_dataset, Dataset
from SLTP_column_name_versions import ColumnVersions

### Kew WCVP: https://kew.iro.bl.uk/concern/datasets/32f77ea6-0f7b-4b2d-b7b3-173ed4ca2d6a?locale=en
"""
When getting data from online portals, use utf-8 whenever you can. 
Downstream formatting might specify utf-8, so ISO or Latin-1 may cause bugs or breaks


SLTPvA:
    - Alpaca prompting format
    -
    - JSON format:
        {
        "catalogNumber": "",
        "order": "",
        "family": "",
        "scientificName": "",
        "scientificNameAuthorship": "",
        "genus": "",
        "subgenus": "",
        "specificEpithet": "",
        "verbatimTaxonRank": "",
        "infraspecificEpithet": "",
        "identifiedBy": "",
        "recordedBy": "",
        "recordNumber": "",
        "verbatimEventDate": "",
        "habitat": "",
        "occurrenceRemarks": "",
        "associatedTaxa": "",
        "country": "",
        "stateProvince": "",
        "county": "",
        "municipality": "",
        "locality": "",
        "decimalLatitude": "",
        "decimalLongitude": "",
        "verbatimCoordinates": "",
        "minimumElevationInMeters": "",
        "maximumElevationInMeters": ""
        }

Example dataset name:
    phyloforfun/HLT_MICH_Angiospermae_SLTPvA_v1.0_OCR-C25-L25-E50-R10

    Project:
        hf_domain = 'HLT'
            'HLT' - "herbarium label transcription"
        hf_institution = 'MICH'
            Herbarium code
        hf_input_dataset = 'Angiospermae'
            One word description of the parent dataset
        hf_SLTP_version = 'SLTPvA'
            The JSON keys used. Defined above
        hf_input_dataset_version = 'v1.0'
            Version of the parent dataset used. For example, v1.0 could be pulled from April, while v1.1 could be pulled from December and have new records
    Synthetic OCR augmentations
        hf_OCR_cap_C = '25'
            25% chance that any given cell will become ALL CAPS
        hf_OCR_cap_L = '25'
            25% chance that any given cell will become all lower case
        hf_OCR_error_rate = '50'
            50% chance that a row will have simulated OCR errors introduced
        hf_OCR_char_error_rate = '10'
            10% chance that a character within a cell will be ['substitute', 'insert', 'delete'] based on predefined, common OCR errors

"""
class DataImporter:
    # Class property for SLTPvA columns
    # taxonomy_columns = ['family','genus','specificEpithet','infraspecificEpithet','scientificName','scientificNameAuthorship','verbatimTaxonRank', ]
    # wcvp_columns = ['family','genus','specificepithet','infraspecificepithet','scientfiicname','scientfiicnameauthorship','taxonrank',]
    
    # MICH_to_SLTPvA_columns = ['catalogNumber', 'order', 'family', 
    #                   'scientificName', 'scientificNameAuthorship', 'genus', 'subgenus', 
    #                   'specificEpithet', 'verbatimTaxonRank', 'infraspecificEpithet', 
    #                   'identifiedBy', 'recordedBy', 'recordNumber', 'verbatimEventDate', 
    #                   'habitat', 'occurrenceRemarks', 'associatedTaxa', 'country', 'stateProvince', 'county', 
    #                   'municipality', 'locality', 'decimalLatitude', 'decimalLongitude', 
    #                   'verbatimCoordinates', 'minimumElevationInMeters', 'maximumElevationInMeters']
    
    # MICH_to_SLTPvA_json = {
    #     "catalogNumber": "",
    #     "order": "",
    #     "family": "",
    #     "scientificName": "",
    #     "scientificNameAuthorship": "",
    #     "genus": "",
    #     "subgenus": "",
    #     "specificEpithet": "",
    #     "verbatimTaxonRank": "",
    #     "infraspecificEpithet": "",
    #     "identifiedBy": "",
    #     "recordedBy": "",
    #     "recordNumber": "",
    #     "verbatimEventDate": "",
    #     "habitat": "",
    #     "occurrenceRemarks": "",
    #     "associatedTaxa": "",
    #     "country": "",
    #     "stateProvince": "",
    #     "county": "",
    #     "municipality": "",
    #     "locality": "",
    #     "decimalLatitude": "",
    #     "decimalLongitude": "",
    #     "verbatimCoordinates": "",
    #     "minimumElevationInMeters": "",
    #     "maximumElevationInMeters": "",
    #     }
    
    # taxonomy_json = {
    #     "family": "",
    #     "genus": "",
    #     "specificEpithet": "",
    #     "infraspecificEpithet": "",
    #     "scientificName": "",
    #     "scientificNameAuthorship": "",
    #     "verbatimTaxonRank": "",
    #     }
    

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

        self.CV = ColumnVersions()
        self.MICH_to_SLTPvA_columns = self.CV.get_MICH_to_SLTPvA_columns()
        self.MICH_to_SLTPvA_json = self.CV.get_MICH_to_SLTPvA_json()

        self.taxonomy_columns = self.CV.get_taxonomy_columns()
        self.taxonomy_json = self.CV.get_taxonomy_json()

        self.wcvp_columns = self.CV.get_wcvp_columns()
        self.wcvp_json = self.CV.get_wcvp_json()


    def MICH_to_SLTPvA(self):
        SLTP_version = 'SLTPvA'
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for encoding in encodings:
            try:
                df = dd.read_csv(self.csv_file_path, assume_missing=True, usecols=self.MICH_to_SLTPvA_columns, dtype=str, encoding=encoding)
                return df, SLTP_version
            except:
                pass
        raise ValueError(f"Failed to read the CSV file with the tried encodings: {encodings}")
    

    def WCVP_to_SLPTvA(self):
        SLTP_version = 'SLTPvA_WCVP'
        try:
            df = dd.read_csv(
                self.csv_file_path, 
                sep='|',
                assume_missing=True,
                dtype=str,
            )

            df = df.rename(columns=dict(zip(self.wcvp_columns, self.taxonomy_columns)))
            # Selecting only the columns listed in DataImporter.taxonomy_columns
            df = df[self.taxonomy_columns]

            return df, SLTP_version
        except Exception as e:
            print(f"An error occurred {e}")
            raise 


    def save_SLTP_version(self, df, SLTP_version):
        # Splitting the file path to add "_SLTPvA" before the file extension
        file_name, file_extension = os.path.splitext(self.csv_file_path)
        new_file_path = f"{file_name}_{SLTP_version}{file_extension}"

        # Saving the DataFrame to the new file path
        with TqdmCallback(desc=f"Saving {SLTP_version} CSV"):
            df.to_csv(new_file_path, single_file=True, index=False)


class DataAugmentation:
    def __init__(self, df, column_names, csv_file_path):
        self.df = df
        self.column_names = column_names
        self.csv_file_path = csv_file_path

    @staticmethod
    def _dropout_augmentation_partition(df, column_names):
        MAX_N_DROPOUT = max(5, len(column_names) - 5)
        augmented_rows = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Augmenting Rows --- Dropout"):
            for ind_drop in range(MAX_N_DROPOUT):  # Generate 10 augmented versions for each row
                new_row = row.copy()
                drop_cols = random.sample(column_names, ind_drop+1)
                new_row[drop_cols] = ''
                augmented_rows.append(new_row)
        return pd.DataFrame(augmented_rows)
    
    def dropout_augmentation(self):
        # n_columns_to_drop = random.randint(1, int(0.8 * len(self.column_names)))
        augmented_df = self.df.map_partitions(
            lambda df: self._dropout_augmentation_partition(df, self.column_names),
            meta=self.df._meta
        )
        # Concatenate the original and augmented dataframes
        with TqdmCallback(desc="Concat Augmented Data"):
            combined_df = dd.concat([self.df, augmented_df])
        return combined_df

    def save_SLTP_dropout_version(self, df, SLTP_version):
        # Compute and save the DataFrame
        if isinstance(df, dd.DataFrame):
            with TqdmCallback(desc="Computing DataFrame"):
                df = df.compute()
            file_name, file_extension = os.path.splitext(self.csv_file_path)
            new_file_path = f"{file_name}_{SLTP_version}_dropout{file_extension}"
            with TqdmCallback(desc="Saving SLTPvA_dropout CSV"):
                df.to_csv(new_file_path, index=False)
            return new_file_path
        else:
            raise TypeError("The provided data is neither a Dask DataFrame nor a Dask Series.")
        

class AlpacaDatasetCreator:
    SLTPvA_instructions_into = "Refactor the unstructured text into a valid JSON dictionary. The key names follow the Darwin Core Archive Standard. If a key lacks content, then insert an empty string. Fill in the following JSON structure as required: "
    SLTPvA_WCVP_instructions_into = "The JSON object provided outlines an organism's taxonomic hierarchy. Your task is to validate the accuracy of this taxonomy. Ensure each taxonomic rank is correctly assigned; if not, relocate values to their proper place. Also, be aware of minor typographical errors like insertions, deletions, or substitutions in the data, and correct these as needed. Complete the task by filling in the JSON structure accordingly: "
    
    # CASE_C = 0.25 # Chance that a cell is ALL CAPS
    # CASE_L = 0.25 # Chance that a cell is all lower case
    # OCR_ERROR_RATE = 0.25 # Chance that a whole row will get subjected to OCR errors
    # CHAR_ERROR_RATE = 0.05 # Chance that an individual character will undergo an error (for a row that is in OCR_ERROR_RATE)

    def __init__(self, importer, csv_file_path, JSON_structure, version, CASE_C, CASE_L, OCR_ERROR_RATE, CHAR_ERROR_RATE, TAXONOMY_SHUFFLE_RATE = None):
        self.CASE_C = CASE_C
        self.CASE_L = CASE_L
        self.OCR_ERROR_RATE = OCR_ERROR_RATE
        self.CHAR_ERROR_RATE = CHAR_ERROR_RATE
        self.TAXONOMY_SHUFFLE_RATE = TAXONOMY_SHUFFLE_RATE
        self.version = version

        if self.version == 'SLTPvA':
            cols = importer.MICH_to_SLTPvA_columns
        elif self.version == 'SLTPvA_WCVP':
            cols = importer.taxonomy_columns

        try:
            encodings = ['utf-8', 'latin1', 'ISO-8859-1']
            for encoding in encodings:
                try:
                    self.df = dd.read_csv(csv_file_path, assume_missing=True, usecols=cols, dtype=str, encoding=encoding)
                except Exception as e:
                    print(f"{e}")
        except:
            raise ValueError(f"Failed to read the CSV file with the tried encodings: {encodings}")
    
        self.JSON_structure = JSON_structure

        if self.version == 'SLTPvA':
            self.formatted_instruction = AlpacaDatasetCreator.SLTPvA_instructions_into + json.dumps(self.JSON_structure)
        elif self.version == 'SLTPvA_WCVP':
            self.formatted_instruction = AlpacaDatasetCreator.SLTPvA_WCVP_instructions_into + json.dumps(self.JSON_structure)
        else:
            raise

    def generate_dataset(self, output_file_path, sample_size=1000):
        total_rows = self.df.shape[0].compute()
        sample_fraction = sample_size / total_rows
        if sample_size != -1:
            sampled_df = self.df.sample(frac=sample_fraction, random_state=2023)
        else:
            sampled_df = self.df

        if self.version == 'SLTPvA':
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for _, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0].compute(), desc=f"Sampling {sample_size} rows from whole dataset"):
                    formatted_entry = {
                        'instruction': self.formatted_instruction,
                        'input': self.create_synthetic_OCR_input(row),
                        'output': self.create_output_json(row).strip()
                    }
                    f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')

        elif self.version == 'SLTPvA_WCVP':
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for _, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0].compute(), desc=f"Sampling {sample_size} rows from whole dataset"):
                    formatted_entry = {
                        'instruction': self.formatted_instruction,
                        'input': self.create_synthetic_taxonomy(row),
                        'output': self.create_output_json(row).strip()
                    }
                    f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
        else:
            raise

    def create_output_json(self, row):
        output_json = {}
        for key in self.JSON_structure.keys():
            value = row[key] if key in row and not pd.isna(row[key]) else ""
            output_json[key] = value
        return json.dumps(output_json)

    def create_synthetic_OCR_input(self, row):
        augmented_row_values = []

        apply_ocr_errors = random.random() < self.OCR_ERROR_RATE  # 50% chance to apply OCR errors

        for value in row.values:
            if not pd.isna(value):
                augmented_value = self.augment_capitalization(value)
                if apply_ocr_errors:
                    augmented_value = self.simulate_ocr_insertion(augmented_value)
                    augmented_value = self.simulate_ocr_errors(augmented_value)
                augmented_row_values.append(augmented_value)
            else:
                augmented_row_values.append("")

        random.shuffle(augmented_row_values)
        return ' '.join(map(str, augmented_row_values))

    def create_synthetic_taxonomy(self, row):
        augmented_row_json = {}
        values_list = []

        apply_ocr_errors = random.random() < self.OCR_ERROR_RATE  # 50% chance to apply OCR errors
        shuffle_values = random.random() < self.TAXONOMY_SHUFFLE_RATE  # 20% chance to shuffle values

        for key in self.JSON_structure.keys():
            value = row[key] if key in row and not pd.isna(row[key]) else ""
            augmented_value = self.augment_capitalization(value)
            if apply_ocr_errors:
                augmented_value = self.simulate_ocr_errors(augmented_value)
            values_list.append(augmented_value)

        if shuffle_values:
            random.shuffle(values_list)

        for key, value in zip(self.JSON_structure.keys(), values_list):
            augmented_row_json[key] = value

        return json.dumps(augmented_row_json)
    
    def augment_capitalization(self, value):
        rand_num = random.random()  # Generates a random number between 0 and 1
        if rand_num < self.CASE_C:
            # Convert to all caps with probability P
            return value.upper()
        elif rand_num < self.CASE_C + self.CASE_L:
            # Convert to lowercase with probability L
            return value.lower()
        else:
            # Keep the value as is
            return value
    
    def simulate_ocr_insertion(self, value):
        # Simulates the insertion of a random string before or after the value.
        random_string = self.random_string(random.randint(1, 5))
        if random.random() < 0.5:
            # Insert random string before the value
            return " " + random_string + " "  + value
        else:
            # Insert random string after the value
            return value + " " + random_string + " " 
        
    def random_string(self, length):
        # Generates a random string of a given length.
        return ''.join([self.random_character() for _ in range(length)])
        
    def simulate_ocr_errors(self, value):
        # Simulates OCR errors in the given text value.
        characters = list(value)
        i = 0
        while i < len(characters):
            if random.random() < self.CHAR_ERROR_RATE:
                error_type = random.choice(['substitute', 'insert', 'delete'])
                if error_type == 'substitute':
                    characters[i] = self.random_substitution(characters[i])
                    i += 1
                elif error_type == 'insert':
                    characters.insert(i, self.random_character())
                    i += 2  # Skip the inserted character
                elif error_type == 'delete':
                    characters.pop(i)
                    # Don't increment i, as we want to check the next character that shifts into the current position
            else:
                i += 1

        return ''.join(characters)

    def random_substitution(self, char):
        substitutions = {
            'O': '0', '0': 'O', 'o': '0', 'D': 'O', 'd': 'o',
            'I': '1', '1': 'I', 'i': '1', 'l': '1', 'L': '1', 
            'l': '|', '1': '|', 'i': '|', 'I': '|',
            'E': '3', 'e': '3', 'F': 'E',
            'A': '4', 'a': '4', 'R': 'A',
            'S': '5', 's': '5', 'Z': '2', 'z': '2',
            'G': '6', 'g': '6', 'C': 'G', 'c': 'g',
            'T': '7', 't': '7', 'J': 'T', 'j': 't',
            'B': '8', 'b': '8', 'P': 'B', 'p': 'b',
            'Q': '9', 'q': '9', 'g': 'q',
            'U': 'V', 'u': 'v', 'V': 'U', 'v': 'u',
            'n': 'm', 'M': 'N', 'N': 'M',
            'f': 't', 'h': 'b', 'Y': 'V',
            'x': 'k', 'K': 'X', 'W': 'M',
        }
        return substitutions.get(char, char)

    def random_character(self):
        return random.choice(" ABCDEF GHIJKLMN OPQR STUVWXYZ 01234 56789 abcdef ghijkl mnopqrs tuvwxyz `~!@#$%^&*()_+|:?><,./;[]=-")

    

    def get_token_from_yaml(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                return data.get('hf_token')  # Assuming the token is stored under the key 'token'
        except Exception as e:
            raise Exception(f"Error reading token from YAML file: {e}")
                
    def upload_to_huggingface(self, output_file_path, dataset_name):
        # Load the dataset file into a Hugging Face Dataset
        dataset = load_dataset('json', data_files=output_file_path, split='train')

        # Push to Hugging Face Hub
        dataset.push_to_hub(dataset_name, private=False)

'''
# Example usage
if __name__ == '__main__':
    dir_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_path = os.path.join(dir_home,'PRIVATE.yaml')

    size_map = {'tiny': 100,
                'small': 1000,
                'medium': 10000,
                'large': 100000,
                'xlarge': 1000000,
                'full': -1,}


    # ### Upload to HF
    # hf_domain = 'HLT'
    # hf_institution = 'MICH'
    # hf_input_dataset = 'Angiospermae'
    # hf_input_dataset_version = 'v1-0'
    # hf_SLTP_version = 'SLTPvA'
    # hf_size = 'medium'

    # hf_OCR_cap_C = '25'
    # hf_OCR_cap_L = '25'
    # hf_OCR_error_rate = '50'
    # hf_OCR_char_error_rate = '05'
    # if hf_OCR_cap_C and hf_OCR_cap_L and hf_OCR_error_rate and hf_OCR_char_error_rate:
    #     dataset_name = f"{hf_domain}_{hf_institution}_{hf_input_dataset}_{hf_SLTP_version}_{hf_input_dataset_version}_{hf_size}__OCR-C{hf_OCR_cap_C}-L{hf_OCR_cap_L}-E{hf_OCR_error_rate}-R{hf_OCR_char_error_rate}"
    # else: # For no OCR augmentation
    #     dataset_name = f"{hf_domain}_{hf_institution}_{hf_input_dataset}_{hf_SLTP_version}_{hf_input_dataset_version}_{hf_size}"
    # print(dataset_name)

    # # Convert to decimal for probability
    # CASE_C = int(hf_OCR_cap_C) / 100.0
    # CASE_L = int(hf_OCR_cap_L) / 100.0
    # OCR_ERROR_RATE = int(hf_OCR_error_rate) / 100.0
    # CHAR_ERROR_RATE = int(hf_OCR_char_error_rate) / 100.0

    # N_ROWS = size_map.get(hf_size)


    # csv_file_path = 'D:/Dropbox/SLTP/datasets/MICH_Angiospermae/occurrences.csv'
    # output_json_file_path = f'D:/Dropbox/SLTP/datasets/MICH_Angiospermae/{dataset_name}.json'

    

    ### Run these once per new occurrences.csv, then you can use AlpacaDatasetCreator with the occurrences_SLTPvA_dropout.csv
    # importer = DataImporter(csv_file_path)
    # original_df, SLTP_version = importer.MICH_to_SLTPvA()
    # importer.save_SLTP_version(original_df, SLTP_version)

    # augmenter = DataAugmentation(original_df, DataImporter.MICH_to_SLTPvA_columns, csv_file_path)
    # combined_df = augmenter.dropout_augmentation()
    # csv_file_path_dropout = augmenter.save_SLTP_dropout_version(combined_df, SLTP_version)
    
    # csv_file_path_dropout = 'D:/Dropbox/SLTP/datasets/MICH_Angiospermae/occurrences_SLTPvA_dropout.csv'
    # creator = AlpacaDatasetCreator(csv_file_path_dropout, DataImporter.MICH_to_SLTPvA_json, SLTP_version, CASE_C, CASE_L, OCR_ERROR_RATE, CHAR_ERROR_RATE)
    # creator.generate_dataset(output_json_file_path, sample_size=N_ROWS)

    

    # creator.upload_to_huggingface(output_json_file_path, dataset_name)



    ##### Build taxonomy correction dataset
    hf_domain = 'HLT'
    hf_institution = 'Kew'
    hf_input_dataset = 'WCVP'
    hf_input_dataset_version = 'v1-0'
    hf_SLTP_version = 'SLTPvA'
    hf_size = 'full'

    hf_OCR_cap_C = '25'
    hf_OCR_cap_L = '25'
    hf_OCR_error_rate = '50'
    hf_OCR_char_error_rate = '10'
    hf_taxonomy_shuffle_rate = '20'
    if hf_OCR_cap_C and hf_OCR_cap_L and hf_OCR_error_rate and hf_OCR_char_error_rate and not hf_taxonomy_shuffle_rate:
        dataset_name = f"{hf_domain}_{hf_institution}_{hf_input_dataset}_{hf_SLTP_version}_{hf_input_dataset_version}_{hf_size}__OCR-C{hf_OCR_cap_C}-L{hf_OCR_cap_L}-E{hf_OCR_error_rate}-R{hf_OCR_char_error_rate}"
    elif hf_OCR_cap_C and hf_OCR_cap_L and hf_OCR_error_rate and hf_OCR_char_error_rate and hf_taxonomy_shuffle_rate:
        dataset_name = f"{hf_domain}_{hf_institution}_{hf_input_dataset}_{hf_SLTP_version}_{hf_input_dataset_version}_{hf_size}__T{hf_taxonomy_shuffle_rate}-OCR-C{hf_OCR_cap_C}-L{hf_OCR_cap_L}-E{hf_OCR_error_rate}-R{hf_OCR_char_error_rate}"
    else: # For no OCR augmentation
        dataset_name = f"{hf_domain}_{hf_institution}_{hf_input_dataset}_{hf_SLTP_version}_{hf_input_dataset_version}_{hf_size}"
    print(dataset_name)

    # Convert to decimal for probability
    CASE_C = int(hf_OCR_cap_C) / 100.0
    CASE_L = int(hf_OCR_cap_L) / 100.0
    OCR_ERROR_RATE = int(hf_OCR_error_rate) / 100.0
    CHAR_ERROR_RATE = int(hf_OCR_char_error_rate) / 100.0
    TAXONOMY_SHUFFLE_RATE = int(hf_taxonomy_shuffle_rate) / 100.0

    N_ROWS = size_map.get(hf_size)

    WCVP_file_path = 'D:/Dropbox/SLTP/datasets/WCVP/wcvp_taxon.csv'
    WCVP_SLTPvA_file_path = 'D:/Dropbox/SLTP/datasets/WCVP/wcvp_taxon_SLTPvA_WCVP.csv'
    WCVP_output_json_file_path = f'D:/Dropbox/SLTP/datasets/WCVP/{dataset_name}.json'

    # importer = DataImporter(WCVP_file_path)
    # df, SLTP_version = importer.WCVP_to_SLPTvA()
    # importer.save_SLTP_version(df, SLTP_version)
    SLTP_version = 'SLTPvA_WCVP' #REMOVE
    creator = AlpacaDatasetCreator(WCVP_SLTPvA_file_path, DataImporter.taxonomy_json, SLTP_version, CASE_C, CASE_L, OCR_ERROR_RATE, CHAR_ERROR_RATE, TAXONOMY_SHUFFLE_RATE)
    creator.generate_dataset(WCVP_output_json_file_path, sample_size=N_ROWS)
    creator.upload_to_huggingface(WCVP_output_json_file_path, dataset_name)


'''