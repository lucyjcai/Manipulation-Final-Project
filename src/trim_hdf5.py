import h5py
import numpy as np
import sys
from pathlib import Path

# ---------- USER INPUT ----------
input_files = [f"teleop_data/sim_open_drawer/episode_{i}_label{j}.hdf5" for i in range(30) for j in [31, 53, 62]]
# --------------------------------


def get_length(f):
    """Get the trajectory length T from one HDF5 file."""
    with h5py.File(f, "r") as h:
        lens = []
        observations = h["observations"]
        # actions = h["actions"]
        # print("actions", type(actions))
        for key in ["qpos", "qvel"]:
            lens.append(len(observations[key]))
            print(key, len(observations[key]))
        return min(lens)


# 1. Find smallest T across all files
lengths = [get_length(f) for f in input_files]
T_min = min(lengths)

print("=== Trajectory lengths ===")
for f, L in zip(input_files, lengths):
    print(f"{f}: {L}")
print(f"\n--> Minimum length = {T_min}\n")


# 2. Process each file by truncating to T_min
def trim_file(input_path, T_min):
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_trimmed.hdf5")

    print(f"Trimming {input_path} → {output_path}")

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        actions = src["action"][:]
        observations = src["observations"]
        qpos = observations["qpos"][:]
        qvel = observations["qvel"][:]
        images = observations["images"]
        camera0 = images["camera0"][:]
        camera1 = images["camera1"][:]
        camera2 = images["camera2"][:]

        dst.attrs["sim"] = True

        dst.create_dataset("action", data=actions[:T_min], compression="gzip")
        obs_grp = dst.create_group("observations")
        obs_grp.create_dataset("qpos", data=qpos[:T_min], compression="gzip")
        obs_grp.create_dataset("qvel", data=qvel[:T_min], compression="gzip")

        img_grp = obs_grp.create_group("images")
        img_grp.create_dataset("camera0", data=camera0[:T_min], compression="gzip")
        img_grp.create_dataset("camera1", data=camera1[:T_min], compression="gzip")
        img_grp.create_dataset("camera2", data=camera2[:T_min], compression="gzip")

    return output_path


# output_paths = [trim_file(f, T_min) for f in input_files]

# print("\nDone!")
# print("Trimmed files:")
# for p in output_paths:
#     print(" -", p)

print(f"\nUnified sequence length: {T_min}")
