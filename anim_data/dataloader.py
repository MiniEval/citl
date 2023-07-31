from itertools import repeat

from anim_data.motion import MotionData
import glob
import numpy as np
import multiprocessing as mp


class DataLoader:
    def __init__(self, bvh_dir, excluded_joints=None, seed=None, min_sample_length=120):
        datafiles = glob.glob(bvh_dir + "/**/*.bvh", recursive=True)
        self.manager = mp.Manager()
        self.pool = mp.Pool(16)
        self.data = self.manager.list()
        self.excluded_joints = excluded_joints
        if self.excluded_joints is None:
            self.excluded_joints = []

        for i, file in enumerate(datafiles):
            print("\rLoading data %d / %d" % (i + 1, len(datafiles)), end="")
            self.data.append(open(file, "r").read())

        print(" Done!")

        self.motions = self.manager.list([MotionData() for _ in range(len(self.data))])
        self.seeds = np.random.SeedSequence(seed)

        # self._lazy_load(self.motions, self.data, 0)

        self.pool.starmap(self._lazy_load, zip([self.motions for _ in range(len(self.data))],
                                               [self.data for _ in range(len(self.data))],
                                               [i for i in range(len(self.data))],
                                               repeat(self.excluded_joints)))

        self.motions = list(self.motions)
        self.motions = [m for m in self.motions if m.duration >= min_sample_length]

        n = 0
        for m in self.motions:
            n += m.duration
        print(n, "frames loaded in dataset")

    @staticmethod
    def _lazy_load(motions, data, i: int, exclude=None):
        if motions[i].duration == 0:
            motion = motions[i]
            motion.load_from_data(data[i], quaternion=True, exclude=exclude)
            motions[i] = motion
            data[i] = None

        return motions[i]

    @staticmethod
    def _random_trim(motion, sample_length: int, rng):
        start = rng.integers(motion.duration - sample_length)
        end = start + sample_length
        sample = motion.clip(start, end)
        return sample

    def get_samples(self, n: int, sample_length: int):
        seeds = self.seeds.spawn(n+1)
        rng = [np.random.default_rng(s) for s in seeds]
        idx = rng[-1].integers(len(self.motions), size=n)
        samples = []
        for i in range(n):
            samples.append(self._random_trim(self.motions[idx[i]], sample_length, rng[i]))

        return samples

    def get_hierarchy(self):
        motion = self._lazy_load(self.motions, self.data, 0)
        return motion.get_hierarchy()

    def restart_pool(self):
        self.pool.close()
        self.pool = mp.Pool(mp.cpu_count() // 2)
