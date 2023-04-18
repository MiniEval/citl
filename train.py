import math
import random

import sys
import torch

from anim_data.dataloader import DataLoader
from models.inpainter import Inpainter


class DirectorTrainer:
    def __init__(self, sample_length, batch_size, dataset, dataset_path):
        self.sample_length = sample_length
        self.batch_size = batch_size
        self.device = torch.device("cuda")

        excluded_joints = None
        self.p_scale = 1
        if dataset == 'cmu':
            excluded_joints = ["LeftToeBase_end", "RightToeBase_end", "Head_end", "LeftToeBase", "RightToeBase",
                               "LeftFingerBase", "LeftHandIndex1", "LeftHandIndex1_end", "LThumb", "LThumb_end",
                               "RightFingerBase", "RightHandIndex1", "RightHandIndex1_end", "RThumb", "RThumb_end"]
            self.p_scale = 2.54 / 0.45 / 64.50309367
        elif dataset == 'lafan':
            excluded_joints = ["LeftToe_end", "RightToe_end", "Head_end", "LeftHand_end", "RightHand_end"]
            self.p_scale = 0.02

        self.data = DataLoader(dataset_path, excluded_joints)

        joints = self.data.get_hierarchy().shape[0]
        offsets = torch.tensor(self.data.motions[0].get_offsets(), dtype=torch.float32,
                               device=self.device) * self.p_scale

        self.inpainter = Inpainter(embed_size=512, joints=joints, hierarchy=self.data.get_hierarchy(),
                                   offsets=offsets, features=7, max_length=120,
                                   heads=8, key_layers=8, interm_layers=8, dec_layers=8, dropout=0, device=self.device)
        self.inpainter.train()

        self.lr = 0.004
        self.steps = 1
        self.warmup_steps = 1000
        self.decay_steps = 1000
        self.optim = torch.optim.Adam(params=self.inpainter.parameters(), lr=self.lr)

    def _update_lr(self):
        lr = self.lr * min(math.pow(self.steps, -0.5), self.steps * math.pow(self.warmup_steps, -1.5))

        for g in self.optim.param_groups:
            g['lr'] = lr

    def get_samples(self, batches, frames):
        samples = self.data.get_samples(batches, frames)
        n_keys = random.randint(frames // 24, frames // 4)

        input_tensors = []
        keyframes = []

        for sample in samples:
            data_p = torch.tensor(sample.get_global_location_data(), dtype=torch.float32, device=self.device).unsqueeze(0)
            data_q = torch.tensor(sample.get_rotation_data(), dtype=torch.float32, device=self.device).unsqueeze(0)

            input_tensors.append(torch.cat((data_p, data_q), dim=-1))

            # random keyframes
            perm = torch.randperm(self.sample_length - 2)[:n_keys] + 1
            keyframes.append(perm)

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
        # _motions = torch.reshape(motions, (batches, frames, -1))

        keyframe_idx = torch.reshape(keyframes, (*keyframes.shape, 1, 1)).repeat(1, 1, *motions.shape[-2:]).to(self.device)

        # keyframe_idx = torch.repeat_interleave(keyframes.unsqueeze(-1), _motions.shape[-1], dim=-1).to(self.device)
        keyposes = torch.gather(motions, 1, keyframe_idx).clone().contiguous()

        return keyposes, keyframes, motions[..., :3], motions[..., 3:], glob_q

    def train(self, epochs):
        for ep in range(1, epochs+1):
            self._update_lr()
            self.optim.zero_grad()

            with torch.no_grad():
                keyposes, keyframes, real_p, real_q, real_glob_q = self.get_samples(self.batch_size, self.sample_length)

            output_p, output_q, output_glob_q = self.inpainter.evaluate(keyposes, keyframes, frames=self.sample_length, normalise_output_q=False)

            l_pos = torch.mean(torch.sum(torch.abs(output_p - real_p), dim=-1))
            l_root = torch.mean(torch.sum((torch.abs(output_p[:, :, 0] - real_p[:, :, 0])), dim=-1))
            l_quat = torch.mean(torch.sum((torch.abs(output_q - real_q)), dim=-1))
            l_quat_global = torch.mean(torch.sum((torch.abs(output_glob_q - real_glob_q)), dim=-1))

            global_mul = max(0.0, min(self.steps / 2000 - 1.0, 1.0))
            loss = l_quat + l_root + (l_pos + l_quat_global) * global_mul

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.inpainter.parameters(), 1.0)
            self.optim.step()

            with torch.no_grad():
                l2p = torch.mean(torch.sqrt(torch.sum((output_p - real_p) ** 2, dim=(-2, -1))))
                l2q = torch.mean(torch.sqrt(torch.sum((output_glob_q - real_glob_q) ** 2, dim=(-2, -1))))

                print("#%d - L_pos: %.4f; L_root: %.4f; L_quat: %.4f; L_quat_global: %.4f; L2P: %.4f; L2Q: %.4f" %
                      (ep, l_pos.item(), l_root.item(), l_quat.item(), l_quat_global.item(), l2p.item(), l2q.item()))

            if ep % 100 == 0:
                self.inpainter.save_models("./saves/Epoch%d.pt" % ep)
                self.data.restart_pool()

            self.steps += 1


if __name__ == "__main__":
    dataset = sys.argv[1]
    dataset_path = sys.argv[2]

    trainer = DirectorTrainer(sample_length=120, batch_size=64, dataset=dataset, dataset_path=dataset_path)
    trainer.train(30000)
