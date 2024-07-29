## Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from constants import CAMERA_NAMES, TEXT_EMBEDDINGS, FILE_COUNTS
import random
import glob 
from logger import logger
import pdb

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

class EpisodicDatasetRobopen(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats,num_episodes):
        super(EpisodicDatasetRobopen).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None
        self.h5s = []
        self.trials = []
        self.task_emb_per_trial = []
        self.verbose = True
        self.h5s = {}
        lens = []
        
        files = list()
        randfiles = list()
        n = 100
        
        # dataset_dir : /data/Datasets/
        for i in glob.glob("{}/*/".format(dataset_dir)):
            print(i)
            randfiles.append(sorted(glob.glob(os.path.join(i, "*.hdf5"))))

        for i in randfiles:
            l = len(i)
            num = int(n/250 * l)
            for file in range(num):
                files.append(i[file])

        files = sorted(files)

        # files = sorted(glob.glob(os.path.join(dataset_dir + "*/*/", '*.h5')))
        for filename in files:
            #for 20 tasks hardcoded, modify as needed
            if 'Pick_the_pumpkin' in filename:
                task_emb = TEXT_EMBEDDINGS[0]
            elif 'Pick_two_cubes_using_single_arm' in filename:
                task_emb = TEXT_EMBEDDINGS[1]
            elif 'Pick_two_cubes_with_two_arms_separately' in filename:
                task_emb = TEXT_EMBEDDINGS[2]            
            elif 'Play_the_chess' in filename:
                task_emb = TEXT_EMBEDDINGS[3]
            else:
                # task_emb = TEXT_EMBEDDINGS[0]
                'SINGLE TASK embedding wont be used'
                continue # modifyed, except dir "stir" etc.

            h5 = h5py.File(filename, 'r')
            for key, trial in h5.items():
                if(trial['data']['time'].shape[0] != 42):
                    continue
                # Open the trial and extract metadata
                lens.append(trial['data']['ctrl_arm'].shape[0])
                # Bookkeeping for all the trials
                self.trials.append(trial)
                self.task_emb_per_trial.append(task_emb)

        self.trial_lengths = np.cumsum(lens)
        self.max_idx = self.trial_lengths[-1]
        print("TOTAL TRIALS",len(self.trials))
        self.trials = self.trials[:num_episodes]

        assert self.num_episodes == len(self.trials) ## sanity check that all files are loaded, remove if needed

        print('TOTAL TRIALS = num_episodes = ',len(self.trials))
        self.__getitem__(0)


    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        sample_full_episode = False # hardcode
        trial_idx = self.episode_ids[idx]
        trial = self.trials[trial_idx]
        task_emb = self.task_emb_per_trial[trial_idx]
        camera_names = CAMERA_NAMES

        action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1).astype(np.float32)
        original_action_shape = action.shape
        cutoff = 2 #10#5 
        episode_len = original_action_shape[0] -cutoff ## cutoff last few

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        qpos = trial['data']['qp_arm'][start_ts].astype(np.float32)
        qvel = trial['data']['qv_arm'][start_ts].astype(np.float32)

        image_dict = dict()
        for cam_name in camera_names:
            image_dict[cam_name] = trial['data'][f'{cam_name}'][start_ts]
        # get all actions after and including start_ts
        action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1)[max(0, start_ts - 1):].astype(np.float32) # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned 

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action[:-cutoff]
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        task_emb = torch.from_numpy(np.asarray(task_emb)).float()

        return image_data, qpos_data, action_data, is_pad, task_emb


class ImprovedEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base, num_episodes=None, max_episode_len=800):
        super(ImprovedEpisodicDataset, self).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.num_episodes = num_episodes
        self.max_episode_len = max_episode_len
        self.is_sim = None
        self.trials = []
        self.file_counts = []
        self.task_emb_per_trial = []
        self.files = []
        self.verbose = True
        
        self._load_episodes()
        self.__getitem__(0)  # initialize self.is_sim

    def _load_episodes(self):
        _, self.files = get_files(self.dataset_dir, self.num_episodes, n=200) # len(files) = 100
        self.trials = []
        self.task_emb_per_trial = []
        lens = []
        file_sum = 0
        
        for filename in self.files:
            if 'Pick_the_pumpkin' in filename:
                task_emb = TEXT_EMBEDDINGS[0]
            # elif 'Pick_two_cubes_using_single_arm' in filename:
            #     task_emb = TEXT_EMBEDDINGS[1]
            elif 'Pick_two_cubes_with_two_arms_separately' in filename:
                task_emb = TEXT_EMBEDDINGS[2]
            elif 'Play_the_chess' in filename:
                task_emb = TEXT_EMBEDDINGS[3]
            else:
                print(f'SINGLE TASK embedding wont be used for {filename}')
                continue
            try:
                with h5py.File(filename, 'r') as root:
                    # if 'data' not in trial or 'time' not in trial['data'] or trial['data']['time'].shape[0] != 42:
                    #     continue
                    print(f"load_episode: {filename}")

                    # self.trial.apppend(root) # TODO: trial可以提前加载进来, 也就是这个load_episode的意义。这里修改了,下面需要连带着变换。
                    self.file_counts.append(filename)
                    # origin: lens.append(trial['data']['ctrl_arm'].shape[0]) (entire aciton = 'ctrl_arm' + 'ctrl_ee')
                    lens.append(root['action'][()].shape[0]) 
                    # Bookkeeping for all the trials
                    # self.trials.append(trial)  # temp deal!
                    self.task_emb_per_trial.append(task_emb)
            except OSError as e:
                print(f"Error occurred when loading file {filename}: {e}")
                continue
                
            
            

        # ! not understand why do this
        self.trial_lengths = np.cumsum(lens)
        self.max_idx = self.trial_lengths[-1] if lens else 0

        if self.verbose and self.file_counts == self.num_episodes:
            print(f"TOTAL TRIALS: {len(self.file_counts)}")
            print(f"TOTAL TRIALS = num_episodes = {len(self.file_counts)}")

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        file_load_path = self.files[idx] # data_path
        logger.info(f"file {file_load_path} is loading..") 
        
        # TODO: 1. _load_episode, to own all trials 2. trial[idx] access every trial data.
        with h5py.File(file_load_path, 'r') as trial:
            is_sim = trial.attrs['sim']
            is_compress = trial.attrs['compress']
            original_action_shape = trial['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode
            episode_len = min(original_action_shape[0], self.max_episode_len) # max_episode_len: 800
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            # 随机选择起始时间点
            start_ts = np.random.choice(episode_len)
            
            # 获取状态和动作
            qpos = trial['/observations/qpos'][start_ts]
            actions = trial['/observations/qpos'][start_ts+1:episode_len]
            # TODO: or actions = root['/observations/qpos'][1:], waiting to check aciton dim
            if len(actions) > 0:
                actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            else:
                print("actions not exits.")
                # 重复上一次动作
                # actions = np.repeat(actions[-1][np.newaxis, :], episode_len - start_ts, axis=0)
                # 置0
                actions = np.zeros((episode_len - start_ts, 14))
                # continue

            if self.use_robot_base:
                qpos = np.concatenate((qpos, trial['/base_action'][start_ts]), axis=0)
                base_actions = trial['/base_action'][start_ts:episode_len]
                actions = np.concatenate((actions, base_actions), axis=1)

            # 处理图像
            image_dict = {}
            image_depth_dict = {}
            for cam_name in list(self.camera_names)[0]:
                # pdb.set_trace()
                if is_compress:
                    decoded_image = trial[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = trial[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = trial[f'/observations/images_depth/{cam_name}'][start_ts]

        # 填充动作序列
        action_len = episode_len - start_ts
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        # pdb.set_trace()
        padded_action[:action_len] = actions
        
        # 创建填充标记
        action_is_pad = np.zeros(episode_len)
        action_is_pad[action_len:] = 1

        # 处理图像数据
        all_cam_images = [image_dict[cam_name] for cam_name in list(self.camera_names)[0]]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        # 处理深度图像数据（如果使用）
        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = [image_depth_dict[cam_name] for cam_name in list(self.camera_names)[0]]
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            image_depth_data = image_depth_data / 255.0

        # 归一化状态和动作
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        action_data = torch.from_numpy(padded_action).float()
        action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        action_is_pad = torch.from_numpy(action_is_pad).bool()

        self.is_sim = is_sim
        # pdb.set_trace()
        task_emb = torch.from_numpy(np.asarray(self.task_emb_per_trial[idx])).float() # check true.(have the same path to file_load_path )
        
        # return already check!
        # return image_data, image_depth_data, qpos_data, action_data, action_is_pad, task_emb
        return image_data, qpos_data, action_data, action_is_pad, task_emb


def get_norm_stats_robopen(dataset_dir,num_epsiodes):
    # files = []
    # for directory in dataset_dir:
    #     files.append()
    files = list()
    randfiles = list()
    n = 100
    
    # dataset_dir : /data/Datasets/
    for i in glob.glob("{}/*/".format(dataset_dir)):
        print(i)
        randfiles.append(sorted(glob.glob(os.path.join(i, "*.hdf5"))))

    for i in randfiles:
        l = len(i)
        num = int(n/250 * l)
        for file in range(num):
            files.append(i[file])

    files = sorted(files)

    print('files',files)
    all_qpos_data = []
    all_action_data = []
    cutoff = 2 #10#5 
    for filename in files:
        # Check each file to see how many entires it has
        h5 = h5py.File(filename, 'r')
        # with h5py.File(filename, 'r') as h5:
        for key, trial in h5.items():
            # Open the trial and extract metadata
            qpos = trial['data']['qp_arm'][()].astype(np.float32)
            qvel = trial['data']['qv_arm'][()].astype(np.float32)
            camera_names = CAMERA_NAMES
            action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1).astype(np.float32)


            if(trial['data']['time'].shape[0] != 42):
                continue
            all_qpos_data.append(torch.from_numpy(qpos[:-cutoff]))
            all_action_data.append(torch.from_numpy(action[:-cutoff]))
            # if len(qpos)==41:
                # all_qpos_data.append(torch.from_numpy(qpos[:-(cutoff-1)]))
                # all_action_data.append(torch.from_numpy(action[:-(cutoff-1)]))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

def get_norm_stats_improved(dataset_dir, num_episodes):
    # 限制处理的文件数量
    _, files_selected = get_files(dataset_dir, num_episodes, n=200) # len: num_episode
    logger.info(f"Select files num:{len(files_selected)}")
    assert len(files_selected) == 142 # temp
    
    # progress： 2024.7.23，18:40
    all_qpos_data = []
    all_action_data = []
    for filename in files_selected: 
        print("filename: ", filename)
        if filename == '/data/Datasets/Play_the_chess/Play_the_chess_episode_000.hdf5':
            print(f"file:{filename} is skipped.")
            continue
        with h5py.File(filename, 'r') as root: 
            # TODO: if condition(...), continue
            # pdb.set_trace()
            

            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            print("qpos shape:", qpos.shape)

            # 添加数据到列表中
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    # 将列表转换为张量
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # 计算动作数据的均值和标准差
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clamp(action_std, 1e-2, 10)  # 裁剪标准差

    # 计算qpos数据的均值和标准差
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clamp(qpos_std, 1e-2, 10)  # 裁剪标准差

    # 创建统计信息字典
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos[0]  # 使用第一个qpos作为示例
    }

    return stats

def load_data_improved(dataset_dir, num_episodes, arm_delay_time, use_depth_image,
              use_robot_base, camera_names, batch_size_train, batch_size_val):
     # obtain train test split
    train_ratio = 0.8 # change as needed
    file_num_last, _ = get_files(dataset_dir, num_episodes, n=200)
    print(f"load data file num last is {file_num_last}.")
    shuffled_indices = np.random.permutation(file_num_last)
    train_indices = shuffled_indices[:int(train_ratio * file_num_last)]
    val_indices = shuffled_indices[int(train_ratio * file_num_last):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_improved(dataset_dir, num_episodes)
    logger.info("Get norm stats done.")
    # construct dataset and dataloader
    # pdb.set_trace()
    train_dataset = ImprovedEpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)
    val_dataset = ImprovedEpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)
    logger.info(f"train dataset load done. Train length is {len(train_dataset)}")
    logger.info(f"val dataset load done.Val length is {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, collate_fn=improved_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, collate_fn=improved_collate_fn)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_data(dataset_dir, num_episodes, batch_size_train, batch_size_val):
    # obtain train test split
    train_ratio = 0.8 # change as needed
    # shuffled_indices = np.random.permutation(num_episodes)
    # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    shuffled_indices = np.random.permutation(FILE_COUNTS)
    train_indices = shuffled_indices[:int(train_ratio * FILE_COUNTS)]
    val_indices = shuffled_indices[int(train_ratio * FILE_COUNTS):]
    
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_robopen(dataset_dir, num_episodes)
    # construct dataset and dataloader
    train_dataset = ImprovedEpisodicDataset(train_indices, dataset_dir, norm_stats,num_episodes)
    val_dataset = ImprovedEpisodicDataset(val_indices, dataset_dir, norm_stats,num_episodes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



### helper functions

def improved_collate_fn(batch):
    max_sequence_length = 600  # 这个值可以根据您的具体情况调整
    
    # 分离批次中的不同组件
    image_data, qpos_data, action_data, action_is_pad, task_emb = zip(*batch)
    
    # 获取这个批次中的最大序列长度
    max_len = max(a.size(0) for a in action_data)
    
    # 填充序列，但只填充到这个批次中的最大长度
    action_data_padded = pad_sequence(action_data, batch_first=True, padding_value=0)
    action_is_pad_padded = pad_sequence(action_is_pad, batch_first=True, padding_value=True)
    
    # 如果最大长度小于某个阈值，我们可以保持原样
    # 如果大于阈值，我们可以随机裁剪或者分割序列    
    if max_len > max_sequence_length:
        # 随机选择一个起始点，裁剪序列
        start = torch.randint(0, max_len - max_sequence_length + 1, (1,)).item()
        action_data_padded = action_data_padded[:, start:start+max_sequence_length]
        action_is_pad_padded = action_is_pad_padded[:, start:start+max_sequence_length]

    # 堆叠其他组件
    image_data = torch.stack(image_data)
    qpos_data = torch.stack(qpos_data)
    task_emb = torch.stack(task_emb)
    
    return {
        'image_data': image_data,
        'qpos_data': qpos_data,
        'action_data': action_data_padded,
        'action_is_pad': action_is_pad_padded,
        'task_emb': task_emb
    }

def get_files(dataset_dir, num_episodes, n=100):
    """
    从数据集目录中获取文件列表, 由于sort过, 每次取的文件均一致。
    
    Args:
        dataset_dir (str): 数据集目录路径。
        n (int): 从每个子文件夹中选取的文件数量。默认为 100。
        num_episodes (int): 限制处理的文件数量。如果为 None,则处理所有文件。
    
    Returns:
        list: 选取的文件路径列表。
    """
    files = list()
    randfiles = list()
    print(f"dataset_dir :{dataset_dir}") 
    
    # 遍历数据集目录的子文件夹,获取 HDF5 文件路径
    for i in glob.glob(f"{dataset_dir[0]}/*/"):
        print("data subfolder:", i)
        randfiles.append(sorted(glob.glob(os.path.join(i, "*.hdf5"))))
    print("randfiles length:", len(randfiles))
    assert len(randfiles)==3, f"Expected 3 randfiles, but got {len(randfiles)}."
    
    # 从每个子文件夹中选取部分文件
    for i in randfiles:
        l = len(i)
        num = int(n/250 * l)
        for file in range(num):
            files.append(i[file])
    
    files = sorted(files)  
    print("GET FILE NUMS :", len(files))
    # global FILE_COUNTS
    # FILE_COUNTS = len(files)
    file_length_last = len(files)
     
    # 限制处理的文件数量
    files = files[:num_episodes]
    
    return file_length_last, files


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
