import pickle
import os

from datetime import datetime


def generate_time_suffix():
    return datetime.now().strftime("%y%m%d_%H%M%S")


def create_backup(*objects):

    folder_name = "../data/backups"
        
    time_suffix = generate_time_suffix() 
    
    folder_name = os.path.join(folder_name, time_suffix)
    os.makedirs(folder_name)
    
    backup_filename = os.path.join(folder_name, "backup.pickle")
    with open(backup_filename, 'wb') as f:
        pickle.dump(objects, f)
        
    return time_suffix
        

def recover_backup(time_suffix):
    
    folder_name = "../data/backups"

    folder_name = os.path.join(folder_name, time_suffix)
    
    backup_filename = os.path.join(folder_name, "backup.pickle")
    with open(backup_filename, 'rb') as f:
        objects = pickle.load(f)
        
    return objects