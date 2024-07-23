# conda activate aloha

import os 
import glob
import sys
sys.path.append("../")
from constants import TEXT_EMBEDDINGS
import h5py

dataset_dir = "/data/Datasets/"
files = list()
randfiles = list()
n = 100
for i in glob.glob("{}/*/".format(dataset_dir)):
    # print(i)
    randfiles.append(sorted(glob.glob(os.path.join(i, "*.hdf5"))))

for i in randfiles:
    l = len(i)
    num = int(n/250 * l)
    for file in range(num):
        files.append(i[file])

files = sorted(files)        
# print("files: ", files)
print("files length: ", len(files))
# assert len(files) == 194

in_file = list()
for filename in files:
    #for 20 tasks hardcoded, modify as needed
    if 'Pick_the_pumpkin' in filename:
        task_emb = TEXT_EMBEDDINGS[0]
        in_file.append(filename)
    elif 'Pick_two_cubes_using_single_arm' in filename:
        task_emb = TEXT_EMBEDDINGS[1]
        in_file.append(filename)
    elif 'Pick_two_cubes_with_two_arms_separately' in filename:
        task_emb = TEXT_EMBEDDINGS[2] 
        in_file.append(filename)           
    elif 'Play_the_chess' in filename:
        task_emb = TEXT_EMBEDDINGS[3]
        in_file.append(filename)
    else:
        task_emb = TEXT_EMBEDDINGS[0]
                
    'SINGLE TASK embedding wont be used'

    print(f"filename :{filename}")
    h5 = h5py.File(filename, 'r')
    for key, trial in h5.items():
        print(key, trial)
        print(h5['action'].shape)
    



    # if(trial['data']['time'].shape[0] != 42):
    #     continue
    # # Open the trial and extract metadata
    # lens.append(trial['data']['ctrl_arm'].shape[0])
    # # Bookkeeping for all the trials
    # self.trials.append(trial)
    # self.task_emb_per_trial.append(task_emb)


# print("file list: ", len(in_file))