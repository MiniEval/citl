import math
import random

import sys

import numpy as np
import torch

from anim_data.dataloader import DataLoader
from anim_data.npss import compute_npss
from models.inpainter import Inpainter


class CITLEval:
    def __init__(self, sample_length, dataset, test_dataset_path, model_ckpt, seed=0, n_interval=5):
        self.sample_length = sample_length
        self.n_interval = n_interval
        self.device = torch.device("cuda")

        excluded_joints = None
        self.p_scale = 1
        if dataset == 'cmu':
            excluded_joints = ["LeftToeBase_end", "RightToeBase_end", "Head_end", "LeftToeBase", "RightToeBase",
                               "LeftFingerBase", "LeftHandIndex1", "LeftHandIndex1_end", "LThumb", "LThumb_end",
                               "RightFingerBase", "RightHandIndex1", "RightHandIndex1_end", "RThumb", "RThumb_end"]
            self.p_scale = 0.056444
        elif dataset == 'lafan':
            excluded_joints = ["LeftToe_end", "RightToe_end", "Head_end", "LeftHand_end", "RightHand_end"]
            self.p_scale = 0.02

        self.data = DataLoader(test_dataset_path, excluded_joints, min_sample_length=sample_length)
        self.keys = [i for i in range(len(self.data.motions))]

        joints = self.data.get_hierarchy().shape[0]
        offsets = torch.tensor(self.data.motions[0].get_offsets(), dtype=torch.float32,
                               device=self.device) * self.p_scale

        self.inpainter = Inpainter(embed_size=512, joints=joints, hierarchy=self.data.get_hierarchy(),
                                   offsets=offsets, features=7, max_length=128,
                                   heads=8, key_layers=8, interm_layers=8, dec_layers=8, dropout=0, device=self.device)
        self.inpainter.load_models(model_ckpt)
        self.inpainter.train()

        torch.manual_seed(seed)
        random.seed(seed)

    def get_samples(self, frames):
        key_idx = random.randint(0, len(self.keys) - 1)
        key = self.keys[key_idx]
        start = random.randint(1, self.data.motions[key].duration - frames)
        samples = [self.data.get_sample(key, start, start + frames)]

        input_tensors = []
        keyframes = []

        for sample in samples:
            data_p = torch.tensor(sample.get_global_location_data(), dtype=torch.float32, device=self.device).unsqueeze(0)
            data_q = torch.tensor(sample.get_rotation_data(), dtype=torch.float32, device=self.device).unsqueeze(0)

            input_tensors.append(torch.cat((data_p, data_q), dim=-1))

            # uniform keyframes
            i = 0
            indices = [0]
            while i + self.n_interval < self.sample_length - 1:
                i += self.n_interval
                indices.append(i)
            indices.append(self.sample_length - 1)
            keyframes.append(torch.tensor(indices, dtype=torch.int64))

        motions = torch.cat(input_tensors, dim=0).contiguous()
        motions[..., :3] = motions[..., :3] * self.p_scale

        glob_p, glob_q = self.inpainter._fk(motions[..., 3:], motions[..., 0, :3])
        motions[..., :3] = glob_p

        keyframes = torch.stack(keyframes, dim=0)
        keyframes[..., 0] = 0
        keyframes[..., -1] = self.sample_length - 1
        keyframes, _ = torch.sort(keyframes, dim=-1)

        keyframe_idx = torch.repeat_interleave(keyframes.unsqueeze(-1), 3, dim=-1).to(self.device)
        roots = torch.gather(motions[..., 0, :3], 1, keyframe_idx)

        root_means = torch.mean(roots, dim=1, keepdim=True).unsqueeze(-2)
        motions[..., :3] -= root_means

        keyframe_idx = torch.reshape(keyframes, (*keyframes.shape, 1, 1)).repeat(1, 1, *motions.shape[-2:]).to(self.device)

        keyposes = torch.gather(motions, 1, keyframe_idx).clone().contiguous()

        return keyposes, keyframes, motions[..., :3], motions[..., 3:], glob_q

    def eval(self, n_tests):
        l2p_list = list()
        l2q_list = list()
        npss_list = list()
        for ep in range(n_tests):
            with torch.no_grad():
                keyposes, keyframes, real_p, real_q, real_glob_q = self.get_samples(self.sample_length)

                output_p, output_q, output_glob_q = self.inpainter.evaluate(keyposes, keyframes, frames=self.sample_length, normalise_output_q=False)

                idx = torch.cat([torch.arange(self.sample_length), keyframes[0]], dim=0)
                unique, count = torch.unique(idx, return_counts=True)
                inv_indices = unique[count == 1].unsqueeze(0)

                inv_indices = torch.reshape(inv_indices, (1, inv_indices.shape[1], 1, 1))
                inv_indices_p = inv_indices.repeat(1, 1, self.data.get_hierarchy().shape[0], 3).to(self.device)
                inv_indices_q = inv_indices.repeat(1, 1, self.data.get_hierarchy().shape[0], 4).to(self.device)

                pred_glob_p = torch.gather(output_p, 1, inv_indices_p)
                pred_glob_q = torch.gather(output_glob_q, 1, inv_indices_q)

                gt_glob_p = torch.gather(real_p, 1, inv_indices_p)
                gt_glob_q = torch.gather(real_glob_q, 1, inv_indices_q)

                l2p = torch.mean(torch.sqrt(torch.sum((pred_glob_p - gt_glob_p) ** 2, dim=(-2, -1))))
                l2q = torch.mean(torch.sqrt(torch.sum((pred_glob_q - gt_glob_q) ** 2, dim=(-2, -1))))
                npss = compute_npss(torch.cat([real_p, real_glob_q], dim=-1).view(1, self.sample_length, -1).cpu().numpy(),
                                    torch.cat([output_p, output_glob_q], dim=-1).view(1, self.sample_length, -1).cpu().numpy())

                l2p_list.append(l2p.item())
                l2q_list.append(l2q.item())
                npss_list.append(npss.item())

        print("L2P: %.4f, L2Q: %.4f, NPSS: %.4f" % (sum(l2p_list) / n_tests, sum(l2q_list) / n_tests, sum(npss_list) / n_tests))


if __name__ == "__main__":
    dataset = sys.argv[1]
    dataset_path = sys.argv[2]
    model_ckpt = sys.argv[3]

    citl = CITLEval(sample_length=128, dataset=dataset, test_dataset_path=dataset_path, model_ckpt=model_ckpt,
                    seed=10000, n_interval=5)
    citl.eval(50)
