import os
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm
from pprint import pprint


def append_current_timestamp_to_filename(filename):
    """
    Append a timestamp to the filename before the file extension.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base, ext = os.path.splitext(filename)
    return f"{base}_{timestamp}{ext}"

# Modified version that collects and returns results from fn
def do_for_each_file_in_folder(folder_path, fn, file_filter=None):
    """
    Applies the function 'fn' to each file in the given folder,
    collects the returned results in a list, and returns it.
    
    Parameters:
      folder_path (str): Path of the directory containing files.
      fn (callable): A function that takes a file path as its only argument and returns a result (e.g., a dict).
      file_filter (callable, optional): Function that takes a file path and returns a boolean.
    """
    files = [
        filename for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename)) and (file_filter(os.path.join(folder_path, filename)) if file_filter else True)
    ]
    files.sort()
    results = []
    with tqdm(files, unit="file") as pbar:
        for i, filename in enumerate(pbar):
            pbar.set_description(f"Processing file {filename}")
            file_path = os.path.join(folder_path, filename)
            result = fn(file_path)
            results.append(result)

            # Round the float values in the result dictionary
            rounded_result = {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()}

            rounded_result = {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()}         
            print(json.dumps(rounded_result, indent=4))

    return results

def do_and_save_results(folder_path, fn, csv_path=None, file_filter=None):
    """
    Applies the function 'fn' to each file in the given folder, collects the results,
    and saves them as a CSV file.
    
    Parameters:
      folder_path (str): Path of the directory containing files.
      fn (callable): A function that takes a file path as its only argument and returns a result (e.g., a dict).
      csv_path (str): File path where the CSV file will be saved.
      file_filter (callable, optional): Function that takes a file path and returns a boolean.
    """
    results = do_for_each_file_in_folder(folder_path, fn, file_filter)
    
    # If the results are dictionaries, convert them to a DataFrame.
    df = pd.DataFrame(results)
    if csv_path is None:
        csv_path = append_current_timestamp_to_filename("results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def do_for_each_jpeg_file_in_folder(folder_path, fn):
    """
    Applies the function 'fn' to each JPEG file in the given folder.
    
    Parameters:
      folder_path (str): Path of the directory containing JPEG files.
      fn (callable): A function that takes a file path as its only argument and returns a result.
    """
    def jpeg_filter(file_path):
        return file_path.lower().endswith((".jpg", ".jpeg"))
    
    return do_and_save_results(folder_path, fn, file_filter=jpeg_filter)