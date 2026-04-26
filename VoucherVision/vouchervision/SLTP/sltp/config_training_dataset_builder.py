# config.py
import os

def get_config(dataset_domain):
    # Base directory
    dir_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # yaml file containing hf_token
    hf_path = os.path.join(dir_home, 'PRIVATE.yaml')

    # Size map
    size_map = {
        'tiny': 100,
        'small': 1000,
        'medium': 10000,
        'large': 100000,
        'xlarge': 1000000,
        'full': -1,
    }

    # Configuration based on dataset domain
    if dataset_domain == 'taxon':
        dataset = 'WCVP'
        file_name = 'wcvp_taxon.csv'
        sltpva_file_name = 'wcvp_taxon_SLTPvA_WCVP.csv'

        file_path = os.path.join(dir_home, 'datasets', dataset, file_name)
        sltpva_file_path = os.path.join(dir_home, 'datasets', dataset, sltpva_file_name)

        hf_config = {
            'domain': 'TEST', # HLT
            'institution': 'Kew',
            'input_dataset': 'WCVP',
            'input_dataset_version': 'v1-0',
            'SLTP_version': 'SLTPvA_WCVP',
            'size': 'tiny',
            'OCR_cap_C': '25',
            'OCR_cap_L': '25',
            'OCR_error_rate': '50',
            'OCR_char_error_rate': '10',
            'taxonomy_shuffle_rate': '20'
        }

    elif dataset_domain == 'DwC':
        dataset = 'MICH_Angiospermae'
        file_name = 'occurrences.csv'

        file_path = os.path.join(dir_home, 'datasets', dataset, file_name)
        sltpva_file_path = None

        hf_config = {
            'domain': 'TEST',  # HLT
            'institution': 'MICH',
            'input_dataset': 'Angiospermae',
            'input_dataset_version': 'v1-0',
            'SLTP_version': 'SLTPvA',
            'size': 'tiny',
            'OCR_cap_C': '25',
            'OCR_cap_L': '25',
            'OCR_error_rate': '50',
            'OCR_char_error_rate': '05',
            'taxonomy_shuffle_rate': '20'
        }

    else:
        raise ValueError("Invalid dataset domain")

    return {
        "DIR_HOME": dir_home,
        "HF_PATH": hf_path,
        "SIZE_MAP": size_map,
        "DATASET": dataset,
        "FILE_PATH": file_path,
        "SLTPVA_FILE_PATH": sltpva_file_path,
        "HF_CONFIG": hf_config
    }
