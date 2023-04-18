import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_ids = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0

    def get_joints(self, _arg=None):
        joints = []
        j = _arg
        if _arg is None:
            j = self.root

        joints.append(j)
        for c in j.children:
            joints.extend(self.get_joints(c))

        return joints

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []
        channel_idx = 0

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
                    joint_stack[-1].channel_ids.append(channel_idx)
                    channel_idx += 1
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def _recursive_get_frame(self, joint, frame_pose, p, r, order):
        local_rotation = np.zeros(3)
        local_position = np.zeros(3)
        for i, channel in enumerate(joint.channels):
            if channel == "Xposition":
                local_position[0] = frame_pose[joint.channel_ids[i]]
            elif channel == "Yposition":
                local_position[1] = frame_pose[joint.channel_ids[i]]
            elif channel == "Zposition":
                local_position[2] = frame_pose[joint.channel_ids[i]]
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[joint.channel_ids[i]]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[joint.channel_ids[i]]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[joint.channel_ids[i]]

        joint_index = order.index(joint)
        p[joint_index] = local_position
        r[joint_index] = np.deg2rad(local_rotation)

        for c in joint.children:
            self._recursive_get_frame(c, frame_pose, p, r, order)

    def frame_data(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]

        self._recursive_get_frame(self.root, frame_pose, p, r, self.get_joints())
        return p, r

    def all_frame_data(self, start=0, end=-1, step=1):
        if end == -1:
            end = self.frames

        p = np.empty((((end - start - 1) // step) + 1, len(self.joints), 3))
        r = np.empty((((end - start - 1) // step) + 1, len(self.joints), 3))

        for frame in range(0, end - start, step):
            p[frame // step], r[frame // step] = self.frame_data(start + frame)

        return p, r

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))

        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"
