import numpy as np
from anim_data import bvh
from scipy.spatial.transform import Rotation


class MotionData:
    class EulerBoneData:
        def __init__(self, rot_data, loc_data, offset, bone_id=0, parent=None, scale=1.0):
            self.bone_id = bone_id
            self.offset = np.expand_dims(offset, axis=0) * scale

            self.parent = parent

            self.rot = np.zeros((rot_data.shape[0], 3))
            self.loc = np.zeros((rot_data.shape[0], 3))
            self.tf_matrix = None
            self.glob_tf = None
            self.glob_coords = None

            self.set(rot_data, loc_data * scale)

        def set(self, rot, loc, start=None, end=None):
            if start is None or end is None:
                self.rot = rot
                self.loc = loc

                self.tf_matrix = self._get_transform_matrix()
                self.glob_tf = self._get_global_transform()
                self.glob_coords = self._get_global_coords()
            else:
                self.rot[start:end] = rot
                self.loc[start:end] = loc

                self.tf_matrix[start:end] = self._get_transform_matrix(start, end)
                self.glob_tf[start:end] = self._get_global_transform(start, end)
                self.glob_coords[start:end] = self._get_global_coords(start, end)

        def _get_transform_matrix(self, start=0, end=-1):
            if end == -1:
                end = self.rot.shape[0]
            rotation = self.rot[start:end]
            location = self.loc[start:end] + self.offset

            vector = np.zeros((rotation.shape[0], 4, 4), dtype=np.float32)

            r = Rotation.from_euler("ZYX", rotation[:, [2, 1, 0]])
            vector[:, :3, :3] = r.as_matrix()
            vector[:, :3, 3] = location

            vector[:, 3, 3] = 1

            return vector

        def _get_global_transform(self, start=0, end=-1):
            if end == -1:
                end = self.rot.shape[0]
            if self.parent:
                transform = self.parent.glob_tf[start:end]
            else:
                transform = np.reshape(np.eye(4), (1, 4, 4)).repeat(self.loc[start:end].shape[0], axis=0)

            return np.matmul(transform, self.tf_matrix[start:end])

        def _get_global_coords(self, start=0, end=-1):
            if end == -1:
                end = self.rot.shape[0]
            pos_v = np.zeros((self.glob_tf[start:end].shape[0], 4, 1))
            pos_v[:, 3] = 1

            return np.matmul(self.glob_tf[start:end], pos_v).squeeze(2)

    class QuatBoneData:
        def __init__(self, rot_data, loc_data, offset, bone_id=0, parent=None, scale=1.0):
            self.bone_id = bone_id
            self.offset = np.expand_dims(offset, axis=0) * scale

            self.parent = parent

            self.rot = np.zeros((rot_data.shape[0], 4))
            self.rot[:, 0] = 1
            self.loc = np.zeros((rot_data.shape[0], 3))
            self.glob_q = np.zeros(self.rot.shape)
            self.glob_coords = None

            self.set(rot_data, loc_data * scale)

        def set(self, rot, loc, start=None, end=None):
            if start is None or end is None:
                self.rot = rot
                self.rot = self.rot / np.sqrt(np.sum(self.rot ** 2, axis=1, keepdims=True))

                for i in range(self.rot.shape[0] - 1):
                    comp_pos = np.sum(np.abs(self.rot[i+1] - self.rot[i])).item()
                    comp_neg = np.sum(np.abs(self.rot[i+1] + self.rot[i])).item()
                    if comp_pos > comp_neg:
                        self.rot[i+1] *= -1

                self.loc = loc

                self.glob_q = self._get_global_rotation()
                self.glob_coords = self._get_global_coords()
            else:
                self.rot[start:end] = rot
                self.rot[start:end] = self.rot[start:end] / np.sqrt(np.sum(self.rot[start:end] ** 2, axis=1, keepdims=True))
                self.loc[start:end] = loc

                for i in range(self.rot.shape[0] - 1):
                    comp_pos = np.sum(np.abs(self.rot[i+1] - self.rot[i])).item()
                    comp_neg = np.sum(np.abs(self.rot[i+1] + self.rot[i])).item()
                    if comp_pos > comp_neg:
                        self.rot[i+1] *= -1

                self.glob_q[start:end] = self._get_global_rotation(start, end)
                self.glob_coords[start:end] = self._get_global_coords(start, end)

        def _q_mul(self, q1, q2):
            q = np.empty(q1.shape)

            q[:, 0] = (q1[:, 0] * q2[:, 0]) - (q1[:, 1] * q2[:, 1]) - (q1[:, 2] * q2[:, 2]) - (q1[:, 3] * q2[:, 3])
            q[:, 1] = (q1[:, 0] * q2[:, 1]) + (q1[:, 1] * q2[:, 0]) + (q1[:, 2] * q2[:, 3]) - (q1[:, 3] * q2[:, 2])
            q[:, 2] = (q1[:, 0] * q2[:, 2]) - (q1[:, 1] * q2[:, 3]) + (q1[:, 2] * q2[:, 0]) + (q1[:, 3] * q2[:, 1])
            q[:, 3] = (q1[:, 0] * q2[:, 3]) + (q1[:, 1] * q2[:, 2]) - (q1[:, 2] * q2[:, 1]) + (q1[:, 3] * q2[:, 0])

            return q

        def _get_global_rotation(self, start=0, end=-1):
            if end == -1:
                end = self.rot.shape[0]
            if self.parent:
                q1 = self.parent.glob_q[start:end]
                q2 = self.rot[start:end]

                return self._q_mul(q1, q2)
            else:
                return self.rot[start:end]

        def _get_global_coords(self, start=0, end=-1):
            if end == -1:
                end = self.rot.shape[0]
            if self.parent:
                p_coords = self.parent.glob_coords[start:end]
                q = self.parent.glob_q[start:end]
            else:
                p_coords = np.zeros(self.loc[start:end].shape)
                q = np.zeros(self.rot[start:end].shape)
                q[:, 0] = 1

            v = self.loc[start:end] + self.offset
            v = np.reshape(v, (-1, 3))
            qvec = q[:, 1:]
            uv = np.cross(qvec, v, axis=1)
            uuv = np.cross(qvec, uv, axis=1)
            return v + 2 * (q[:, :1] * uv + uuv) + p_coords

    def __init__(self):
        self.bone_data = []
        self.bone_names = []

        self.duration = 0

    def load_from_data(self, data_string, start=0, end=-1, step=1, quaternion=False, exclude=None, scale=1.0):
        data = bvh.Bvh()
        data.parse_string(data_string)

        self.bone_data.clear()
        self.bone_names.clear()
        loc_data, rot_data = data.all_frame_data(start=start, end=end, step=step)

        self.duration = loc_data.shape[0]
        joints = data.get_joints()

        bone_id = 0
        for i, joint in enumerate(joints):
            if exclude is not None:
                if joint.name in exclude:
                    continue
            self.bone_names.append(joint.name)
            parent = None
            if joint.parent:
                parent = self.bone_data[self.bone_names.index(joint.parent.name)]
            if quaternion:
                r = Rotation.from_euler("ZYX", rot_data[:, i, [2, 1, 0]])
                self.bone_data.append(self.QuatBoneData(r.as_quat()[:, [3, 0, 1, 2]], loc_data[:, i], joint.offset, bone_id=bone_id, parent=parent, scale=scale))
            else:
                self.bone_data.append(self.EulerBoneData(rot_data[:, i], loc_data[:, i], joint.offset, bone_id=bone_id, parent=parent, scale=scale))
            bone_id += 1

    def load_from_bvh(self, file, start=0, end=-1, step=1, quaternion=False, exclude=None, scale=1.0):
        with open(file, "r") as f:
            self.load_from_data(f.read(), start, end, step, quaternion, exclude, scale)

    def get_rotation_data(self, start=0, end=-1):
        if end == -1:
            end = self.duration

        rot_data = np.empty((end - start, len(self.bone_data), self.bone_data[0].rot.shape[1]))

        for i, bone in enumerate(self.bone_data):
            rot_data[:, i] = bone.rot[start:end]

        return rot_data

    def get_global_rotation_data(self, start=0, end=-1):
        if end == -1:
            end = self.duration

        rot_data = np.empty((end - start, len(self.bone_data), self.bone_data[0].rot.shape[1]))

        for i, bone in enumerate(self.bone_data):
            rot_data[:, i] = bone.glob_q[start:end]

        return rot_data

    def get_local_location_data(self, start=0, end=-1, include_offset=False):
        if end == -1:
            end = self.duration

        loc_data = np.empty((end - start, len(self.bone_data), 3))

        for i, bone in enumerate(self.bone_data):
            if include_offset:
                loc_data[:, i] = bone.loc[start:end] + bone.offset
            else:
                loc_data[:, i] = bone.loc[start:end]

        return loc_data

    def get_global_location_data(self, start=0, end=-1):
        if end == -1:
            end = self.duration

        loc_data = np.empty((end - start, len(self.bone_data), 3))

        for i, bone in enumerate(self.bone_data):
            loc_data[:, i] = bone.glob_coords[start:end, :3]

        return loc_data

    def get_data(self):
        data = np.empty((self.duration, len(self.bone_data), 7))
        for i, bone in enumerate(self.bone_data):
            data[:, i, :3] = self.bone_data[i].loc
            data[:, i, 3:(self.bone_data[i].rot.shape[1] + 3)] = self.bone_data[i].rot
        return data

    def set_data(self, data, start=None, end=None):
        for i, bone in enumerate(self.bone_data):
            self.bone_data[i].set(data[:, i, 3:(self.bone_data[i].rot.shape[1] + 3)], data[:, i, :3], start, end)
        if start is None and end is None:
            self.duration = data.shape[0]

    def clip(self, start, end):
        new_motion = self.base_copy()
        for i, bone in enumerate(self.bone_data):
            new_motion.bone_data[i].rot = bone.rot[start:end]
            new_motion.bone_data[i].loc = bone.loc[start:end]
            new_motion.bone_data[i].glob_q = bone.glob_q[start:end]
            new_motion.bone_data[i].glob_coords = bone.glob_coords[start:end]
        new_motion.duration = end - start
        return new_motion

    def get_hierarchy(self):
        parents = np.zeros(len(self.bone_data), dtype=int)
        for i, bone in enumerate(self.bone_data):
            if bone.parent is not None:
                parents[i] = bone.parent.bone_id
            else:
                parents[i] = -1
        return parents

    def get_offsets(self):
        offsets = np.zeros((len(self.bone_data), 3))
        for i, bone in enumerate(self.bone_data):
            offsets[i] = bone.offset
        return offsets

    def get_joint(self, name):
        for i, joint in enumerate(self.bone_names):
            if joint == name:
                return self.bone_data[i]
        return None

    def base_copy(self):
        new_motion = MotionData()
        quaternion = isinstance(self.bone_data[0], MotionData.QuatBoneData)
        p = np.array([[0.0, 0.0, 0.0]])
        if quaternion:
            r = np.array([[1.0, 0.0, 0.0, 0.0]])
            for bone in self.bone_data:
                new_motion.bone_data.append(MotionData.QuatBoneData(r, p, bone.offset))
        else:
            for bone in self.bone_data:
                new_motion.bone_data.append(MotionData.EulerBoneData(p, p, bone.offset))

        new_motion.bone_names = list(self.bone_names)
        for i, bone in enumerate(self.bone_data):
            new_motion.bone_data[i].bone_id = bone.bone_id
            if bone.parent is not None:
                new_motion.bone_data[i].parent = new_motion.bone_data[bone.parent.bone_id]

        return new_motion
