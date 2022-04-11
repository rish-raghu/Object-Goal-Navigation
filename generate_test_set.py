import json
from envs import make_vec_envs
from arguments import get_args
import numpy as np
import torch
import os

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    idx_to_name = {
        0: "chair",
        1: "couch",
        2: "potted plant",
        3: "bed",
        4: "toilet",
        5: "tv",
        6: "dining-table",
        7: "oven",
        8: "sink",
        9: "refrigerator",
        10: "book",
        11: "clock",
        12: "vase",
        13: "cup",
        14: "bottle"
    }
    name_to_idx = {name: idx for idx, name in idx_to_name.items()}

    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.set_num_threads(1)
    envs = make_vec_envs(args)

    ep_datas = [[], [], [], []]

    for i in range(200):
        _, infos = envs.reset()
        for e in range(4):
            ep_data = {}
            ep_data['episode_id'] = int(infos[e]['episode_id'])
            ep_data['scene_id'] = "gibson_semantic/{}.glb".format(infos[e]["scene"])
            ep_data['start_position'] = list(infos[e]['sim_pos'])
            ep_data['start_rotation'] = list(infos[e]['sim_rot'])
            ep_data['object_category'] = infos[e]['goal_name']
            ep_data['object_id'] = int(name_to_idx[infos[e]['goal_name']])
            ep_data['floor_id'] = int(infos[e]['floor_idx'])
            ep_datas[e].append(ep_data)
            
    for e, ep_data in enumerate(ep_datas):
        with open('data/datasets/objectnav/gibson/v1.1/val/content/{}_test_episodes.json'.format(
                infos[e]["scene"]), 'w') as f:
            json.dump(ep_data, f)

if __name__ == "__main__":
    main()


