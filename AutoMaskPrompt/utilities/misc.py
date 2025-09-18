import os
import csv
import random
import shutil

def save_log_csv(log_dict:dict, path:str):
    reset_dirs(path)
    with open(os.path.join(path, "score_log.csv"), 'w') as f:
        w = csv.writer(f)
        w.writerow(log_dict.keys())
        w.writerow(log_dict.values())

def reset_dirs(path:str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
