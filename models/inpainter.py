from models.transformers import KeyframeEncoder, IntermediateEncoder, Decoder
import torch


class Inpainter:
    def __init__(self, embed_size, joints, hierarchy, offsets, features=7, max_length=72, heads=8, key_layers=8, interm_layers=8, dec_layers=8, dropout=0.1, device=None):
        self.embed_size = embed_size
        self.joints = joints
        self.features = features
        self.hierarchy = hierarchy
        self.offsets = offsets
        self.max_length = max_length
        self.heads = heads
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.kf_encoder = KeyframeEncoder(embed_size, joints * features, max_length, heads, key_layers, dropout).to(self.device)
        self.interm_encoder = IntermediateEncoder(embed_size, 16, max_length, heads, interm_layers, dropout).to(self.device)
        self.decoder = Decoder(embed_size, 4 * joints + 3, heads, dec_layers, dropout).to(self.device)

    @staticmethod
    def _qmul(q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape

        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    @staticmethod
    def _qrot(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = list(v.shape)
        q = q.view(-1, 4)
        v = v.view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    def _fk(self, local_q, root_p):
        positions_world = []
        rotations_world = []
        expanded_offsets = torch.reshape(self.offsets, (1, 1, *self.offsets.shape)).repeat(local_q.shape[0], local_q.shape[1], 1, 1)
        expanded_offsets = expanded_offsets.contiguous()

        for i in range(len(self.hierarchy)):
            if self.hierarchy[i] == -1:
                positions_world.append(root_p)
                rotations_world.append(local_q[:, :, 0])
            else:
                positions_world.append(self._qrot(rotations_world[self.hierarchy[i]], expanded_offsets[:, :, i]) +
                                       positions_world[self.hierarchy[i]])
                rotations_world.append(self._qmul(rotations_world[self.hierarchy[i]], local_q[:, :, i]))

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)

    def _get_inverse_idx(self, idx, frames):
        # idx: [batches, keyframes], CPU

        inv_indices = []
        for b in range(idx.shape[0]):
            idx_ = torch.cat([torch.arange(frames), idx[b]], dim=0)
            unique, count = torch.unique(idx_, return_counts=True)
            inv_indices.append(unique[count == 1])
        inv_indices = torch.stack(inv_indices, dim=0)

        return inv_indices

    def evaluate(self, keyposes, keyframes, frames=None, normalise_output_q=True):
        # Expect # of keyframes to be equal across all batches
        # keyposes: [batches, keyframes, joints, data] # glob_p & local_q in data
        # keyframes: [batches, keyframes], CPU

        if frames is None:
            frames = keyframes[0, -1] + 1

        batches = keyposes.shape[0]
        # input_q = torch.reshape(keyposes[..., 3:], (batches, keyposes.shape[1], -1, 4))
        # input_p, _ = self._fk(keyposes[..., :3], input_q)
        # input_data = torch.cat([input_p, input_q], dim=-1)
        input_data = torch.reshape(keyposes, (*keyposes.shape[:2], -1))
        interm_frames = self._get_inverse_idx(keyframes, frames)

        kf_enc = self.kf_encoder(input_data, keyframes)
        interm_enc = self.interm_encoder(interm_frames, kf_enc)
        output = self.decoder(interm_enc, interm_frames.to(self.device), kf_enc, keyframes.to(self.device))

        key_scatter_idx = torch.repeat_interleave(keyframes.unsqueeze(-1), output.shape[-1], dim=-1).to(self.device)
        original = torch.cat([keyposes[..., 0, :3], torch.reshape(keyposes[..., 3:], (*keyposes.shape[:2], -1))], dim=-1)
        output = torch.scatter(output, 1, key_scatter_idx, original)

        output_q = torch.reshape(output[..., 3:], (batches, frames, -1, 4))
        output_q_norm = output_q / torch.linalg.vector_norm(output_q, dim=-1, keepdim=True)
        output_p, output_glob_q = self._fk(output_q_norm, output[:, :, :3])

        if normalise_output_q:
            output_q = output_q_norm

        # [batches, frames, joints, 3 (for pos) or 4 (for quat)]
        return output_p, output_q, output_glob_q

    def train(self):
        self.kf_encoder.train()
        self.interm_encoder.train()
        self.decoder.train()

    def eval(self):
        self.kf_encoder.eval()
        self.interm_encoder.eval()
        self.decoder.eval()

    def parameters(self):
        return list(self.kf_encoder.parameters()) + list(self.interm_encoder.parameters()) + list(self.decoder.parameters())

    def save_models(self, filename="model_inpainter.pt"):
        torch.save({
            'kf_encoder': self.kf_encoder.state_dict(),
            'interm_encoder': self.interm_encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }, filename)

    def load_models(self, filename="model_inpainter.pt"):
        checkpoints = torch.load(filename)

        self.kf_encoder.load_state_dict(checkpoints['kf_encoder'])
        self.interm_encoder.load_state_dict(checkpoints['interm_encoder'])
        self.decoder.load_state_dict(checkpoints['decoder'])
