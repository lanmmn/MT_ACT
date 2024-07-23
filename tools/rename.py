import os
import glob

class DatasetRenamer:
    def __init__(self, dataset_dir, task_names=None):
        self.dataset_dir = dataset_dir
        self.task_names = task_names if task_names else ['Pick_the_pumpkin', 'Pick_two_cubes_using_single_arm', 'Pick_two_cubes_with_two_arms_separately', 'Play_the_chess']

    def rename_files(self):
        for task_name in self.task_names:
            task_dir = os.path.join(self.dataset_dir, task_name)
            if os.path.isdir(task_dir):
                for i, filename in enumerate(sorted(glob.glob(os.path.join(task_dir, '*.hdf5')))):
                    new_filename = f"{task_name}_episode_{i:03d}.hdf5"
                    new_filepath = os.path.join(task_dir, new_filename)
                    os.rename(filename, new_filepath)
                    print(f"Renamed {filename} to {new_filepath}")
            else:
                print(f"Directory not found: {task_dir}")

if __name__ == "__main__":
    dataset_dir = "/data/Datasets/" 
    renamer = DatasetRenamer(dataset_dir)
    renamer.rename_files()
