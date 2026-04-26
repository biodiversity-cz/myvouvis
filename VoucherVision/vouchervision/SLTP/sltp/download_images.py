import os, requests, re, time, random, certifi
from dataclasses import dataclass, field
# from difflib import diff_bytes
import pandas as pd
import numpy as np
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from threading import Lock
from concurrent.futures import ThreadPoolExecutor as th
import logging
import http.client as http_client
from utils import validate_dir

# dir_LM2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# dir_current_config = os.path.join(dir_LM2,'configs')
# path_cfg = os.path.join(dir_current_config,'config_download_from_GBIF_all_images_in_file.yml')
# cfg = get_cfg_from_full_path(path_cfg)

# # Run Download
# download_all_images_in_images_csv(cfg)



# TODO The code here that merges occ and img is meant for GBIF, but I added new merging code for MICH. These should be merged 
#      into 1 implementation in the ImageCandidate class 1/13/2024



def download_all_images_in_images_csv(cfg, sampled_csv):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']

    # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,aggOcc_filename,aggImg_filename):
    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)
    
    if cfg['is_custom_file']:
        raise
        # download_from_custom_file(cfg)
    else:
        # Get DWC files
        occ_df, images_df = read_DWC_file(cfg, sampled_csv)

        # Report summary
        print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
        print(images_df.head())
        print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
        print(occ_df.head())

        # Ignore problematic Herbaria
        # if cfg['ignore_banned_herb']:
        #     for banned_url in cfg['banned_url_stems']:
        #         images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]

        ### TODO NEW, needs to match the gbif version
        # Find common 'id' values in both dataframes

        # Ensure ID columns are of the same type
        occ_df['id'] = occ_df['id'].astype(str)
        images_df['coreid'] = images_df['coreid'].astype(str)

        common_ids = set(occ_df['id']).intersection(set(images_df['coreid']))
        print(f"{bcolors.BOLD}Number of common IDs: {len(common_ids)}{bcolors.ENDC}")
        
        # Filter both dataframes to keep only rows with these common 'id' values
        # occ_df_filtered = occ_df[occ_df['id'].isin(common_ids)]
        if len(common_ids) > 0:
            # Filter images_df to keep only rows with these common 'id' values
            images_df = images_df[images_df['coreid'].isin(common_ids)]

            # Report summary
            n_imgs = images_df.shape[0]
            n_occ = occ_df.shape[0]
            print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}Number of occurrence to search through: {n_occ}{bcolors.ENDC}")

            images_df = images_df.drop_duplicates(subset='coreid')
            n_imgs = images_df.shape[0]
            print(f"{bcolors.BOLD}Number of individual images in images file: >>> {n_imgs} <<<{bcolors.ENDC}")
            print(f"{bcolors.BOLD}Will need to randomly select: >>> {n_occ-n_imgs} <<<{bcolors.ENDC}")

            results, failed_cluster = process_image_batch(cfg, images_df, occ_df)

            # Ensure all elements are DataFrames and not None
            successful_downloads_df = [df for df in results if isinstance(df, pd.DataFrame)]
            
            # Check if all DataFrames have the same columns and then concatenate them
            if successful_downloads_df:
                first_df = successful_downloads_df[0]
                if all(df.columns.equals(first_df.columns) for df in successful_downloads_df):
                    successful_downloads_df = pd.concat(successful_downloads_df, ignore_index=True)
                else:
                    raise ValueError("Not all DataFrames in the list have the same columns.")
            else:
                successful_downloads_df = pd.DataFrame()  # or handle the empty case as needed

            return successful_downloads_df, failed_cluster
        else:
            print("No common IDs found between occurrence and images dataframes.")
            return None, None

        

def process_image_batch(cfg, images_df, occ_df):
    futures_list = []
    results = []
    failed_cluster = []
    lock = Lock() 

    # single threaded, useful for debugging
    # for index, image_row in images_df.iterrows():
    #     futures = process_each_image_row( cfg, image_row, occ_df, lock)
    #     futures_list.append(futures)
    # for future in futures_list:
    #     try:
    #         result = future.result(timeout=60)
    #         results.append(result)
    #     except Exception:
    #         results.append(None)


    with th(max_workers=cfg['n_threads']) as executor:
        for index, image_row in images_df.iterrows():
            futures = executor.submit(process_each_image_row, cfg, image_row, occ_df, lock)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result, cluster  = future.result(timeout=60)
                results.append(result)
                failed_cluster.append(cluster)
            except Exception:
                results.append(None)
    return results, failed_cluster



def process_each_image_row(cfg, image_row, occ_df, lock):
    try:
        if 'gbifID' in image_row:
            id_column = 'gbifID'
        # If 'gbifID' is not a key, check if 'id' is a key
        elif 'id' in image_row:
            id_column = 'id'
        elif 'coreid' in image_row:
            id_column = 'coreid'
        else:
            raise ValueError("No suitable ID column found in image_row.")


        print(f"{bcolors.BOLD}Working on image: {image_row[id_column]}{bcolors.ENDC}")
        gbif_id = image_row[id_column]
        gbif_url = image_row['identifier'] 

        occ_row = find_gbifID(gbif_id,occ_df)

        cluster = occ_row['cluster'].iloc[0].astype(int)

        if occ_row is not None:
            # Convert occ_row to DataFrame if it's a Series
            if isinstance(occ_row, pd.Series):
                df_occurrences = occ_row.to_frame().T
            elif isinstance(occ_row, pd.DataFrame) and occ_row.shape[0] == 1:
                df_occurrences = occ_row
            else:
                raise ValueError(f"occ_row has an unexpected structure: {type(occ_row)}")

            # Verify and convert image_row to a single-row DataFrame
            if isinstance(image_row, dict):
                df_multimedia = pd.DataFrame([image_row])
            elif isinstance(image_row, pd.Series):
                df_multimedia = image_row.to_frame().T
            elif isinstance(image_row, pd.DataFrame) and image_row.shape[0] == 1:
                df_multimedia = image_row
            else:
                raise ValueError(f"image_row has an unexpected structure: {type(image_row)}")


            # Get the sets of column names for each DataFrame
            columns_occurrences = set(df_occurrences.columns)
            columns_multimedia = set(df_multimedia.columns)

            # Find and drop the intersection to get shared columns
            shared_columns = columns_occurrences.intersection(columns_multimedia)
            df_multimedia = df_multimedia.drop(columns=shared_columns)

            # Ensure the indices are aligned before concatenation
            df_occurrences.reset_index(drop=True, inplace=True)
            df_multimedia.reset_index(drop=True, inplace=True)

            # Concatenate the DataFrames side by side
            merged_df = pd.concat([df_occurrences, df_multimedia], axis=1)
            print(merged_df)

            ### This is last to avoid downloading an image until *AFTER* the code for merging is successful
            ImageInfo = ImageCandidate(cfg, image_row, occ_row, gbif_url, lock)
            ImageInfo.download_image(lock)

            if ImageInfo.download_success: 
                return merged_df, ""
            else:
                return None, cluster

        else:
            return None, cluster

    except Exception as e:
        print(f"An error occurred in process_each_image_row: {e}")



@dataclass
class ImageCandidate:
    cfg: str = ''
    herb_code: str = '' 
    specimen_id: str = ''
    family: str = ''
    genus: str = ''
    species: str = ''
    fullname: str = ''
    
    cluster: str = ''

    filename_image: str = ''
    filename_image_jpg: str = ''

    url: str = ''
    headers_occ: str = ''
    headers_img: str = ''

    download_success: bool = False

    occ_row: list = field(init=False,default_factory=None)
    image_row: list = field(init=False,default_factory=None)


    def __init__(self, cfg, image_row, occ_row, url, lock):
        # self.headers_occ =  list(occ_row.columns.values)
        # self.headers_img = list(image_row.columns.values)
        self.headers_occ = occ_row
        self.headers_img = image_row
        self.occ_row = occ_row # pd.DataFrame(data=occ_row,columns=self.headers_occ)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)
        self.url = url
        self.cfg = cfg

        self.download_success = False


        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(occ_row)

    # def download_image(self, lock) -> None:
    #     dir_destination = self.cfg['dir_destination_images']
    #     MP_low = self.cfg['MP_low']
    #     MP_high = self.cfg['MP_high']
    #     # Define URL get parameters
    #     sep = '_'
    #     session = requests.Session()
    #     retry = Retry(connect=1) #2, backoff_factor=0.5)
    #     adapter = HTTPAdapter(max_retries=retry)
    #     session.mount('http://', adapter)
    #     session.mount('https://', adapter)

    #     print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
    #     print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
    #     try:
    #         response = session.get(self.url, stream=True, timeout=1.0)
    #         img = Image.open(response.raw)
    #         self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
    #         print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
    #     except Exception as e: 
    #         print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Reason --> {response.reason}{bcolors.ENDC}")
    def download_image(self, lock) -> None:
        http_client.HTTPConnection.debuglevel = 1

        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True



        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        
        # Set up a session with retry strategy
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        session.headers.update(headers)
        session.verify = certifi.where()
        retries = Retry(connect=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        
        

        try:
            response = session.get(self.url, stream=True, timeout=5.0, verify=False)
            response.raise_for_status()  # Check for HTTP errors

            img = Image.open(response.raw)

            was_saved = self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)  # TODO make this occ + img code work for MICH *and* GBIF, right now they are seperate 
            
            if not was_saved:
                raise ImageSaveError(f"Failed to save the image: {self.url}")

            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
            self.download_success = True

        except ImageSaveError as e:
            print(f"{bcolors.FAIL}                {e}{bcolors.ENDC}")

        except requests.exceptions.HTTPError as http_err:
            print(f"{bcolors.FAIL}                HTTP Error --> {http_err}{bcolors.ENDC}")

        except requests.exceptions.ConnectionError as conn_err:
            # Handle connection-related errors, ignore if you don't want to print them
            pass

        except Exception as e:
            # This will ignore the "No active exception to reraise" error
            if str(e) != "No active exception to reraise":
                print(f"{bcolors.FAIL}                SKIP --- No Connection or Rate Limited --> {e}{bcolors.ENDC}")

        finally:
            # Randomized delay
            time.sleep(2 + random.uniform(0, 2))

    def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
        img_mp, img_w, img_h = check_image_size(img)
        was_saved = False
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

        elif MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination,self.filename_image_jpg)
            img.save(image_path)

            #imgSaveName = pd.DataFrame({"image_path": [image_path]})
            # self._add_occ_and_img_data(lock) ########################################################################################## This code works for GBIF occ img, but not MICH directly

            print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
            was_saved = True

        elif img_mp > MP_high:
            if self.cfg['do_resize']:
                [img_w, img_h] = calc_resize(img_w, img_h)
                newsize = (img_w, img_h)
                img = img.resize(newsize)
                image_path = os.path.join(dir_destination,self.filename_image_jpg)
                img.save(image_path)

                #imgSaveName = pd.DataFrame({"imgSaveName": [imgSaveName]})
                # self._add_occ_and_img_data(lock) ########################################################################################## This code works for GBIF occ img, but not MICH directly
                
                print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
                was_saved = True
            else:
                print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")
        return was_saved
    def _add_occ_and_img_data(self, lock) -> None:
        self.image_row = self.image_row.to_frame().transpose().rename(columns={"identifier": "url"}) 
        if 'gbifID' in self.image_row.columns:
            id_column = 'gbifID'
            self.image_row = self.image_row.rename(columns={id_column: 'gbifID_images'}) 
        # If 'gbifID' is not a key, check if 'id' is a key
        elif 'id' in self.image_row.columns:
            id_column = 'id'
            self.image_row = self.image_row.rename(columns={id_column: 'id_images'}) 
        else:
            raise

        new_data = {'fullname': [self.fullname], 'filename_image': [self.filename_image], 'filename_image_jpg': [self.filename_image_jpg]}
        new_data = pd.DataFrame(data=new_data)

        all_data = [new_data.reset_index(), self.image_row.reset_index(), self.occ_row.reset_index()]
        combined = pd.concat(all_data,ignore_index=False, axis=1)

        w_1 = new_data.shape[1] + 1
        w_2 = self.image_row.shape[1] + 1
        w_3 = self.occ_row.shape[1]

        combined.drop([combined.columns[0], combined.columns[w_1], combined.columns[w_1 + w_2]], axis=1, inplace=True)
        headers = np.hstack((new_data.columns.values, self.image_row.columns.values, self.occ_row.columns.values))
        combined.columns = headers
        self._append_combined_occ_image(self.cfg, combined, lock)

    def _append_combined_occ_image(self, cfg, combined, lock) -> None:
        path_csv_combined = os.path.join(cfg['dir_destination_csv'], cfg['filename_combined'])
        with lock:
            try: 
                # Add row once the file exists
                csv_combined = pd.read_csv(path_csv_combined,dtype=str)
                combined.to_csv(path_csv_combined, mode='a', header=False, index=False)
                print(f'{bcolors.OKGREEN}       Added 1 row to combined CSV: {path_csv_combined}{bcolors.ENDC}')

            except Exception as e:
                print(f"{bcolors.WARNING}       Initializing new combined .csv file: [occ,images]: {path_csv_combined}{bcolors.ENDC}")
                combined.to_csv(path_csv_combined, mode='w', header=True, index=False)

class ImageSaveError(Exception):
    """Custom exception for image saving errors."""
    pass

@dataclass
class ImageCandidateCustom:
    cfg: str = ''
    # herb_code: str = '' 
    # specimen_id: str = ''
    # family: str = ''
    # genus: str = ''
    # species: str = ''
    fullname: str = ''

    filename_image: str = ''
    filename_image_jpg: str = ''

    url: str = ''
    # headers_occ: str = ''
    headers_img: str = ''

    # occ_row: list = field(init=False,default_factory=None)
    image_row: list = field(init=False,default_factory=None)


    def __init__(self, cfg, image_row, url, col_name, lock):
        # self.headers_occ =  list(occ_row.columns.values)
        # self.headers_img = list(image_row.columns.values)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)

        self.url = url
        self.cfg = cfg
        self.col_name = col_name

        self.fullname = image_row[col_name]
        self.filename_image = image_row[col_name]
        self.filename_image_jpg = ''.join([image_row[col_name], '.jpg'])
        
        self.download_image(lock)

    def download_image(self, lock) -> None:
        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        # Define URL get parameters
        sep = '_'
        session = requests.Session()
        retry = Retry(connect=1) #2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        try:
            response = session.get(self.url, stream=True, timeout=1.0)
            img = Image.open(response.raw)
            self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
        except Exception as e: 
            print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Reasone --> {response.reason}{bcolors.ENDC}")

    def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

        elif MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination,self.filename_image_jpg)
            img.save(image_path)

            print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")

        elif img_mp > MP_high:
            if self.cfg['do_resize']:
                [img_w, img_h] = calc_resize(img_w, img_h)
                newsize = (img_w, img_h)
                img = img.resize(newsize)
                image_path = os.path.join(dir_destination,self.filename_image_jpg)
                img.save(image_path)

                print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")


def download_from_custom_file(cfg):
    # Get DWC files
    images_df = read_custom_file(cfg)

    col_url = cfg['col_url']
    col_name = cfg['col_name']
    if col_url == None:
        col_url = 'identifier'
    else:
        col_url = col_url

    # Report summary
    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())

    # Ignore problematic Herbaria
    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df[col_url].str.contains(banned_url, na=False)]
    
    # Report summary
    n_imgs = images_df.shape[0]
    print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")

    results = process_custom_image_batch(cfg, images_df)


def read_custom_file(cfg):
    dir_home = cfg['dir_home']
    filename_img = cfg['filename_img']
    # read the images.csv or occurences.csv file. can be txt ro csv
    images_df = ingest_DWC(filename_img,dir_home)
    return images_df

def ingest_DWC(DWC_csv_or_txt_file,dir_home):
    file_path = os.path.join(dir_home, DWC_csv_or_txt_file)
    file_extension = DWC_csv_or_txt_file.split('.')[1]

    try:
        if file_extension == 'txt':
            df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
        elif file_extension == 'csv':
            # Attempt to read with comma separator
            try:
                df = pd.read_csv(file_path, sep=",", header=0, low_memory=False, dtype=str)
            except pd.errors.ParserError:
                try:
                    # If failed, try with a different separator, e.g., semicolon
                    df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
                except:
                    try:
                        df = pd.read_csv(file_path, sep="|", header=0, low_memory=False, dtype=str)
                    except:
                        df = pd.read_csv(file_path, sep=";", header=0, low_memory=False, dtype=str)
        else:
            print(f"{bcolors.FAIL}DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened{bcolors.ENDC}")
            return None
    except Exception as e:
        print(f"Error while reading file: {e}")
        return None

    return df
def process_custom_image_batch(cfg, images_df):
    futures_list = []
    results = []

    lock = Lock() 

    with th(max_workers=13) as executor:
        for index, image_row in images_df.iterrows():
            futures = executor.submit(process_each_custom_image_row, cfg, image_row, lock)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results

def process_each_custom_image_row(cfg, image_row, lock):
    col_url = cfg['col_url']
    col_name = cfg['col_name']

    if col_url == None:
        col_url = 'identifier'
    else:
        col_url = col_url

    gbif_url = image_row[col_url] 

    print(f"{bcolors.BOLD}Working on image: {image_row[col_name]}{bcolors.ENDC}")
    if image_row is not None:
        ImageInfo = ImageCandidateCustom(cfg, image_row, gbif_url, col_name, lock)
    else:
        pass

def calc_resize(w,h):
    if h > w:
        ratio = h/w
        new_h = 5000
        new_w = round(5000/ratio)
    elif w >= h:
        ratio = w/h
        new_w = 5000
        new_h = round(5000/ratio)
    return new_w, new_h

def check_image_size(img):
    [img_w, img_h] = img.size
    img_mp = round(img_w * img_h / 1000000,1)
    return img_mp, img_w, img_h

def check_n_images_in_group(detailedOcc,N):
    fam = detailedOcc['fullname'].unique()
    for f in fam:
        ct = len(detailedOcc[detailedOcc['fullname'].str.match(f)])
        if ct == N:
            print(f"{bcolors.OKGREEN}{f}: {ct}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}{f}: {ct}{bcolors.ENDC}")
def check_image_size(img):
    [img_w, img_h] = img.size
    img_mp = round(img_w * img_h / 1000000,1)
    return img_mp, img_w, img_h

def check_n_images_in_group(detailedOcc,N):
    fam = detailedOcc['fullname'].unique()
    for f in fam:
        ct = len(detailedOcc[detailedOcc['fullname'].str.match(f)])
        if ct == N:
            print(f"{bcolors.OKGREEN}{f}: {ct}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}{f}: {ct}{bcolors.ENDC}")

# Return entire row of file_to_search that matches the gbif_id, else return []
def find_gbifID(gbif_id,file_to_search):
    # Check if 'gbifID' is a key in the DataFrame
    if 'gbifID' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)), :]
        
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['id'].astype(str).str.match(str(gbif_id)), :]
    elif 'coreid' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['coreid'].astype(str).str.match(str(gbif_id)), :]
    # If neither 'gbifID' nor 'id' is a key, raise an error
    else:
        raise KeyError("Neither 'gbifID' nor 'id' found in the column names for the Occurrences file")


    if row_found.empty:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
        return None
    elif 'order' in row_found or  'family' in row_found or  'scientificName' in row_found:
        if row_found['order'].iloc[0] or (row_found['family'].iloc[0]) or (row_found['scientificName'].iloc[0]):
            print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
            return row_found
        else:
            return None
    
    elif 'FamilyName' in row_found or 'GenusName' in row_found or 'SpeciesName' in row_found:
        if row_found['FamilyName'].iloc[0] or (row_found['GenusName'].iloc[0]) or (row_found['SpeciesName'].iloc[0]):
            print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
            return row_found
        else:
            return None

    else:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} successfully found in occurrences file but missing all of 'order' or 'family' or 'scientificName'{bcolors.ENDC}")
        return None
    
def find_ID_custom(gbif_id,file_to_search, col_id):
    # Check if 'gbifID' is a key in the DataFrame
    if col_id in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search[col_id].astype(str).str.match(str(gbif_id)), :]
        
    # # If 'gbifID' is not a key, check if 'id' is a key
    # elif 'id' in file_to_search.columns:
    #     row_found = file_to_search.loc[file_to_search['id'].astype(str).str.match(str(gbif_id)), :]
    # elif 'coreid' in file_to_search.columns:
    #     row_found = file_to_search.loc[file_to_search['coreid'].astype(str).str.match(str(gbif_id)), :]
    # If neither 'gbifID' nor 'id' is a key, raise an error
    else:
        raise KeyError(f"{col_id} not found in the column names for the Occurrences file")
    
    if row_found.empty:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
        return None
    elif (row_found['order'].iloc[0]) or (row_found['family'].iloc[0]) or (row_found['scientificName'].iloc[0]):
        # print(f"########## {row_found['order'].iloc[0]}")
        print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
        return row_found
    else:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} successfully found in occurrences file but missing all of 'order' or 'family' or 'scientificName'{bcolors.ENDC}")
        return None
    
    

def generate_image_filename(occ_row):
    if 'gbifID' in occ_row:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in occ_row:
        id_column = 'id'
    elif 'coreid' in occ_row:
        id_column = 'coreid'
    else:
        raise

    herb_code = remove_illegal_chars(validate_herb_code(occ_row))
    try:
        specimen_id = str(occ_row[id_column].values[0])
        family = remove_illegal_chars(occ_row['family'].values[0])
        genus = remove_illegal_chars(occ_row['genus'].values[0])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet'].values[0]))
    except:
        try:
            specimen_id = str(occ_row[id_column])
            family = remove_illegal_chars(occ_row['family'])
            genus = remove_illegal_chars(occ_row['genus'])
            species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet']))
        except:
            ### MORTON FILES
            specimen_id = str(occ_row[id_column].values[0])
            family = remove_illegal_chars(occ_row['FamilyName'].values[0])
            genus = remove_illegal_chars(occ_row['GenusName'].values[0])
            species = remove_illegal_chars(keep_first_word(occ_row['SpeciesName'].values[0]))
    fullname = '_'.join([family, genus, species])

    filename_image = '_'.join([herb_code, specimen_id, fullname])
    filename_image_jpg = '.'.join([filename_image, 'jpg'])

    return filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname


def validate_herb_code(occ_row):
    possible_keys = ['institutionCode', 'institutionID', 'ownerInstitutionCode', 
                    'collectionCode', 'publisher', 'occurrenceID','MuseumCode']
    # print(occ_row)
    # Herbarium codes are not always in the correct column, we need to find the right one
    # try:
    #     opts = [occ_row['institutionCode'],
    #         occ_row['institutionID'],
    #         occ_row['ownerInstitutionCode'],
    #         occ_row['collectionCode'],
    #         occ_row['publisher'],
    #         occ_row['occurrenceID']]
    #     opts = [item for item in opts if not(pd.isnull(item.values)) == True]
    # except:
    #     opts = [str(occ_row[key]) for key in possible_keys if key in occ_row and not pd.isnull(occ_row[key])]  ######### TODO see if this should be the default
    #     opts = pd.DataFrame(opts)
    #     opts = opts.dropna()
    #     opts = opts.apply(lambda x: x[0]).tolist()
    opts = []
    for key in possible_keys:
        if key in occ_row:
            value = occ_row[key]
            if isinstance(value, pd.Series):
                # Iterate through each element in the Series
                for item in value:
                    if pd.notnull(item) and isinstance(item, str):
                        opts.append(item)
            else:
                # Handle the case where value is not a Series
                if pd.notnull(value) and isinstance(value, str):
                    opts.append(value)

    opts_short = []

    for word in opts:
        #print(word)
        if len(word) <= 8:
            if word is not None:
                opts_short = opts_short + [word]

    if len(opts_short) == 0:
        try:
            herb_code = occ_row['publisher'].values[0].replace(" ","-")
        except:
            try:
                herb_code = occ_row['publisher'].replace(" ","-")
            except:
                herb_code = "ERROR"
    try:
        inst_ID = occ_row['institutionID'].values[0]
        occ_ID = occ_row['occurrenceID'].values[0]
    except:
        try:
            inst_ID = occ_row['institutionID']
            occ_ID = occ_row['occurrenceID']
        
            occ_ID = str(occ_row['occID']) if 'occID' in occ_row and pd.notna(occ_row['occID']) else "" ############## new NOTE
        except:
            try:
                inst_ID = ''
                occ_ID = occ_row['occurrenceID']
            
                occ_ID = str(occ_row['occID']) if 'occID' in occ_row and pd.notna(occ_row['occID']) else "" ############## new NOTE
            except:
                occ_ID = []

    if inst_ID == "UBC Herbarium":
        herb_code = "UBC"
    elif inst_ID == "Naturalis Biodiversity Center":
        herb_code = "L"
    elif inst_ID == "Forest Herbarium Ibadan (FHI)":
        herb_code = "FHI"
    elif 'id.luomus.fi' in occ_ID: 
        herb_code = "FinBIF"
    else:
        if len(opts_short) > 0:
            herb_code = opts_short[0]

    try:
        herb_code = herb_code.values[0]
    except:
        herb_code = herb_code

    # Specific cases that require manual overrides
    # If you see an herbarium DWC file with a similar error, add them here
    if herb_code == "Qarshi-Botanical-Garden,-Qarshi-Industries-Pvt.-Ltd,-Pakistan":
        herb_code = "Qarshi-Botanical-Garden"
    elif herb_code == "12650":
        herb_code = "SDSU"
    elif herb_code == "322":
        herb_code = "SDSU"
    elif herb_code == "GC-University,-Lahore":
        herb_code = "GC-University-Lahore"
    elif herb_code == "Institute-of-Biology-of-Komi-Scientific-Centre-of-the-Ural-Branch-of-the-Russian-Academy-of-Sciences":
        herb_code = "Komi-Scientific-Centre"
    
    return herb_code

def remove_illegal_chars(text):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]","",text)
    return cleaned

def keep_first_word(text):
    if (' ' in text) == True:
        cleaned = text.split(' ')[0]
    else:
        cleaned = text
    return cleaned

def read_DWC_file(cfg, sampled_csv):
    path_DwC_home = cfg['path_DwC_home']
    # filename_occ = cfg['filename_occ']
    filename_img = cfg['filename_img']
    # read the images.csv or occurences.csv file. can be txt ro csv
    # occ_df = ingest_DWC(filename_occ,dir_home)
    images_df = ingest_DWC(filename_img, path_DwC_home)
    return sampled_csv, images_df

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
    CBLUEBG3   = '\33[112m'
