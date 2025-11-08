from pydrake.all import RigidTransform, RollPitchYaw, StartMeshcat
from manipulation.letter_generation import create_sdf_asset_from_letter
import time


if __name__ == "__main__":
    print("Drake import OK.")
    X = RigidTransform(RollPitchYaw(0.1, 0.2, 0.3), [1, 2, 3])
    print("Transform:", X)
    print("Starting MeshCat… (check the VS Code Ports panel)")
    meshcat = StartMeshcat()
    time.sleep(30)
