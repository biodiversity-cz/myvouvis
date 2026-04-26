import os
from config_training_dataset_builder import get_config
from utils_training_dataset_builder import DataImporter, AlpacaDatasetCreator, DataAugmentation

class DatasetBuilderTaxon:
    def __init__(self, config):
        self.config = config

    def _generate_dataset_name(self):
        conf = self.config['HF_CONFIG']
        parts = [conf['domain'], conf['institution'], conf['input_dataset'], conf['SLTP_version'], conf['input_dataset_version'], conf['size']]
        if all(conf[k] for k in ['OCR_cap_C', 'OCR_cap_L', 'OCR_error_rate', 'OCR_char_error_rate', 'taxonomy_shuffle_rate']):
            parts.append(f"T{conf['taxonomy_shuffle_rate']}-OCR-C{conf['OCR_cap_C']}-L{conf['OCR_cap_L']}-E{conf['OCR_error_rate']}-R{conf['OCR_char_error_rate']}")
        elif all(conf[k] for k in ['OCR_cap_C', 'OCR_cap_L', 'OCR_error_rate', 'OCR_char_error_rate']):
            parts.append(f"OCR-C{conf['OCR_cap_C']}-L{conf['OCR_cap_L']}-E{conf['OCR_error_rate']}-R{conf['OCR_char_error_rate']}")
        return "_".join(parts)

    def build(self):
        importer = DataImporter(self.config['SLTPVA_FILE_PATH'])
        dataset_name = self._generate_dataset_name()
        print(dataset_name)

        # Convert to decimal for probability
        conf = self.config['HF_CONFIG']
        case_c = int(conf['OCR_cap_C']) / 100.0
        case_l = int(conf['OCR_cap_L']) / 100.0
        ocr_error_rate = int(conf['OCR_error_rate']) / 100.0
        char_error_rate = int(conf['OCR_char_error_rate']) / 100.0
        taxonomy_shuffle_rate = int(conf['taxonomy_shuffle_rate']) / 100.0

        n_rows = self.config['SIZE_MAP'].get(conf['size'])

        output_json_file_path = os.path.join(self.config['DIR_HOME'],'datasets',self.config['DATASET'],f"{dataset_name}.json")

        # Assuming DataImporter and AlpacaDatasetCreator are defined elsewhere
        creator = AlpacaDatasetCreator(importer, self.config['SLTPVA_FILE_PATH'], importer.taxonomy_json, conf['SLTP_version'], case_c, case_l, ocr_error_rate, char_error_rate, taxonomy_shuffle_rate)
        creator.generate_dataset(output_json_file_path, sample_size=n_rows)
        creator.upload_to_huggingface(output_json_file_path, dataset_name)


class DatasetBuilderDwC:
    def __init__(self, config):
        self.config = config

    def _generate_dataset_name(self):
        conf = self.config['HF_CONFIG']
        parts = [conf['domain'], conf['institution'], conf['input_dataset'], conf['SLTP_version'], conf['input_dataset_version'], conf['size']]
        if all(conf[k] for k in ['OCR_cap_C', 'OCR_cap_L', 'OCR_error_rate', 'OCR_char_error_rate']):
            parts.append(f"OCR-C{conf['OCR_cap_C']}-L{conf['OCR_cap_L']}-E{conf['OCR_error_rate']}-R{conf['OCR_char_error_rate']}")
        return "_".join(parts)

    def build(self):
        dataset_name = self._generate_dataset_name()
        print(dataset_name)

        conf = self.config['HF_CONFIG']
        case_c = int(conf['OCR_cap_C']) / 100.0
        case_l = int(conf['OCR_cap_L']) / 100.0
        ocr_error_rate = int(conf['OCR_error_rate']) / 100.0
        char_error_rate = int(conf['OCR_char_error_rate']) / 100.0

        n_rows = self.config['SIZE_MAP'].get(conf['size'])

        output_json_file_path = os.path.join(self.config['DIR_HOME'], 'datasets', self.config['DATASET'], f"{dataset_name}.json")

        # Data importing and augmentation process
        importer = DataImporter(self.config['FILE_PATH'])
        original_df, SLTP_version = importer.MICH_to_SLTPvA()
        importer.save_SLTP_version(original_df, SLTP_version)

        ### Augmentation needs the dropout file, if it exists, move to next step
        file_name, file_extension = os.path.splitext(self.config['FILE_PATH'])
        check_csv_file_path_dropout = f"{file_name}_{SLTP_version}_dropout{file_extension}"
        if not os.path.exists(check_csv_file_path_dropout):
            augmenter = DataAugmentation(original_df, importer.MICH_to_SLTPvA_columns, self.config['FILE_PATH'])
            combined_df = augmenter.dropout_augmentation()
            csv_file_path_dropout = augmenter.save_SLTP_dropout_version(combined_df, SLTP_version)
        else:
            csv_file_path_dropout = check_csv_file_path_dropout
            print("Dropout file already exists. Moving to the next step.")

        creator = AlpacaDatasetCreator(importer, csv_file_path_dropout, importer.MICH_to_SLTPvA_json, SLTP_version, case_c, case_l, ocr_error_rate, char_error_rate)

        creator.generate_dataset(output_json_file_path, sample_size=n_rows)
        creator.upload_to_huggingface(output_json_file_path, dataset_name)

if __name__ == '__main__':
    # 'taxon' or 'DwC'
    set_type = 'DwC'

    config = get_config(set_type)

    if set_type == 'taxon':
        builder = DatasetBuilderTaxon(config)
        builder.build()
    elif set_type == 'DwC':
        builder = DatasetBuilderDwC(config)
        builder.build()