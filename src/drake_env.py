import numpy as np
from pydrake.all import (
    DiagramBuilder, Simulator, LoadModelDirectives,
    Parser, ImageRgba8U, LeafSystem, BasicVector, Context, ConstantVectorSource
)
from manipulation.station import LoadScenario, MakeHardwareStation


class PositionCommandSystem(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort(
            "q_desired",
            BasicVector(7),
            self.DoOutput
        )
        self.q_desired = np.zeros(7)

    def DoOutput(self, context, output):
        output.SetFromVector(self.q_desired)

class WSGPositionCommandSystem(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorOutputPort(
            "wsg_desired",
            BasicVector(1),
            self.DoOutput
        )
        self.wsg_desired = np.zeros(1)

    def DoOutput(self, context, output):
        output.SetFromVector(self.wsg_desired)

# -------------------------------------------------------------------
# dm_control-style TimeStep
# -------------------------------------------------------------------
class TimeStep:
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward


# -------------------------------------------------------------------
# DrakeEnv that replicates your teleop scene
# -------------------------------------------------------------------
class DrakeEnv:
    def __init__(self, scenario_string, meshcat=None, dt=0.05):
        self.dt = dt

        # ------------------------------------------------------------
        # Build HardwareStation from scenario
        # ------------------------------------------------------------
        scenario = LoadScenario(data=scenario_string)
        self.station = MakeHardwareStation(scenario, meshcat=meshcat)

        builder = DiagramBuilder()
        builder.AddSystem(self.station)

        self.plant = self.station.GetSubsystemByName("plant")
        self.camera_world = self.station.GetSubsystemByName("rgbd_sensor_camera0")
        self.camera_wrist = self.station.GetSubsystemByName("rgbd_sensor_camera1")
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")

        self.position_cmd = builder.AddSystem(PositionCommandSystem())
        builder.Connect(
            self.position_cmd.get_output_port(),
            self.station.GetInputPort("iiwa.position")
        )

        self.wsg_cmd = builder.AddSystem(WSGPositionCommandSystem())
        builder.Connect(
            self.wsg_cmd.get_output_port(),
            self.station.GetInputPort("wsg.position")
        )

        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()

        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.set_target_realtime_rate(1.0)

        # ------------------------------------------------------------
        # ACT interface compatibility
        # ------------------------------------------------------------
        class Task:
            max_reward = 1.0  # ACT requires this attribute
        self.task = Task()

        self.camera_names = ["camera0", "camera1"]

    # -------------------------------------------------------------------
    # Reset the Drake station
    # -------------------------------------------------------------------
    def reset(self):
        self.context = self.diagram.CreateDefaultContext()
        self.simulator = Simulator(self.diagram, self.context)

        # Get plant context
        station_context = self.diagram.GetSubsystemContext(self.station, self.context)
        plant_context = self.station.GetSubsystemContext(self.plant, station_context)

        # Zero initial qpos (change if needed)
        q0 = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
        # --- Set IIWA positions ---
        iiwa_model = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetPositions(
            plant_context,
            iiwa_model,
            q0  # shape must match iiwa num_positions (7)
        )

        # --- Set WSG positions ---
        wsg_model = self.plant.GetModelInstanceByName("wsg")
        self.plant.SetPositions(
            plant_context,
            wsg_model,
            np.array([-0.05, 0.05])  # shape must match wsg num_positions (2 for WSG fingers)
        )

        # Optional: zero velocities
        self.plant.SetVelocities(plant_context, iiwa_model, np.zeros(self.plant.num_velocities(iiwa_model)))
        self.plant.SetVelocities(plant_context, wsg_model, np.zeros(self.plant.num_velocities(wsg_model)))

        # Publish once to initialize sensors
        self.diagram.ForcedPublish(self.context)

        return self._make_timestep(0.0)

    # -------------------------------------------------------------------
    # Step the simulator using ACT action (target qpos)
    # -------------------------------------------------------------------
    def step(self, target_qpos):
        station_context = self.diagram.GetSubsystemContext(self.station, self.context)
        plant_context = self.station.GetSubsystemContext(self.plant, station_context)

        print(target_qpos)
        self.position_cmd.q_desired = np.array(target_qpos[:7])
        self.wsg_cmd.wsg_desired = np.array([target_qpos[-1]])

        # Advance simulation by dt
        t = self.context.get_time()
        self.simulator.AdvanceTo(t + self.dt)

        return self._make_timestep(0.0)

    # -------------------------------------------------------------------
    # ACT expects env._physics.render(...)
    # -------------------------------------------------------------------
    def _physics(self):
        return self

    def render(self, camera_id, height=480, width=640):
        if camera_id == "camera0":
            port = "camera0.rgb_image"
        elif camera_id == "camera1":
            port = "camera1.rgb_image"
        else:
            raise ValueError(f"Unknown camera name: {camera_id}")

        station_context = self.diagram.GetSubsystemContext(self.station, self.context)
        img: ImageRgba8U = self.station.GetOutputPort(port).Eval(station_context)

        # Convert to numpy
        rgba = np.array(np.copy(img.data)).reshape(
            img.height(), img.width(), 4
        )
        rgb = rgba[:, :, :3].astype(np.uint8)

        return rgb

    # -------------------------------------------------------------------
    # Build observation + reward into ACT-compatible TimeStep
    # -------------------------------------------------------------------
    def _make_timestep(self, reward):
        station_context = self.diagram.GetSubsystemContext(self.station, self.context)
        plant_context = self.station.GetSubsystemContext(self.plant, station_context)

        # Extract iiwa and wsg states
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        wsg = self.plant.GetModelInstanceByName("wsg")

        q_iiwa = self.plant.GetPositions(plant_context, iiwa)
        v_iiwa = self.plant.GetVelocities(plant_context, iiwa)

        q_wsg = self.plant.GetPositions(plant_context, wsg)
        v_wsg = self.plant.GetVelocities(plant_context, wsg)

        # Convert wsg distance to width
        wsg_width = q_wsg[1] - q_wsg[0]
        wsg_vel   = v_wsg[1] - v_wsg[0]

        # TODO: if training data normalized gripper values, need to normalize here
        # and also unnormalize in step() when applying policy

        # Final observation format for ACT
        qpos = np.concatenate([q_iiwa, [wsg_width]])
        qvel = np.concatenate([v_iiwa, [wsg_vel]])

        # Capture both cameras
        images = {
            "camera0": self.render("camera0"),
            "camera1": self.render("camera1"),
        }

        obs = dict(qpos=qpos, qvel=qvel, images=images)

        return TimeStep(obs, reward)
