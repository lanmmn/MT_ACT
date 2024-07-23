import torch
import numpy as np
import h5py
import os
import glob
import logging
from improved_episodic_dataset import ImprovedEpisodicDataset, TEXT_EMBEDDINGS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_episodic_dataset():
    # 测试参数
    dataset_dir = "/data/Datasets/"
    camera_names = ["cam1", "cam2"]  # 根据实际情况调整
    norm_stats = {
        "qpos_mean": np.zeros(14),  # 假设qpos长度为14
        "qpos_std": np.ones(14),
        "action_mean": np.zeros(14),
        "action_std": np.ones(14)
    }
    arm_delay_time = 1
    use_depth_image = True
    use_robot_base = True
    num_episodes = 10
    max_episode_len = 500

    # 创建数据集实例
    dataset = ImprovedEpisodicDataset(
        episode_ids=list(range(num_episodes)),
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        arm_delay_time=arm_delay_time,
        use_depth_image=use_depth_image,
        use_robot_base=use_robot_base,
        num_episodes=num_episodes,
        max_episode_len=max_episode_len
    )

    logger.info(f"Dataset created with {len(dataset)} episodes")

    # 测试数据集长度
    assert len(dataset) == num_episodes, f"Dataset length mismatch: {len(dataset)} != {num_episodes}"

    # 测试数据加载
    for i in range(len(dataset)):
        try:
            image_data, image_depth_data, qpos_data, action_data, action_is_pad, task_emb = dataset[i]
            
            # 检查返回值的类型和形状
            assert isinstance(image_data, torch.Tensor), f"image_data is not a torch.Tensor for episode {i}"
            assert isinstance(image_depth_data, torch.Tensor), f"image_depth_data is not a torch.Tensor for episode {i}"
            assert isinstance(qpos_data, torch.Tensor), f"qpos_data is not a torch.Tensor for episode {i}"
            assert isinstance(action_data, torch.Tensor), f"action_data is not a torch.Tensor for episode {i}"
            assert isinstance(action_is_pad, torch.Tensor), f"action_is_pad is not a torch.Tensor for episode {i}"
            assert isinstance(task_emb, torch.Tensor), f"task_emb is not a torch.Tensor for episode {i}"

            assert image_data.shape[0] == len(camera_names), f"Incorrect number of cameras for episode {i}"
            assert image_data.shape[1] == 3, f"Incorrect number of color channels for episode {i}"
            assert action_data.shape[0] == max_episode_len, f"Incorrect action sequence length for episode {i}"
            assert action_is_pad.shape[0] == max_episode_len, f"Incorrect padding mask length for episode {i}"

            logger.info(f"Successfully loaded and verified episode {i}")
        except Exception as e:
            logger.error(f"Error processing episode {i}: {str(e)}")
            raise

    # 测试随机访问
    random_idx = np.random.randint(0, len(dataset))
    try:
        _ = dataset[random_idx]
        logger.info(f"Successfully accessed random episode {random_idx}")
    except Exception as e:
        logger.error(f"Error accessing random episode {random_idx}: {str(e)}")
        raise

    logger.info("All tests passed successfully!")

if __name__ == "__main__":
    test_improved_episodic_dataset()