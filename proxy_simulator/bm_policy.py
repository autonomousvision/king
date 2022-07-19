import torch


class BMActionSequence(torch.nn.Module):
    def __init__(self, args, batch_size, num_agents, sim_horizon):
        """
        Implements a non-reactive driving agent that simply follows a trajectory
        parameterized by a sequence of acceleration and streering actions in
        conjunction with a kinematic bicycle model as a transition model.

        Args:
            batch_size (int): Number of parallel simulations.
            num_agents (int): The number of adversarial agents that move based
                on this policy.
            sim_horizon (int): Number of timesteps to unroll the simulation for.
        """
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.sim_horizon = sim_horizon

        #### TRAJECTORY PARAMETERIZATION ####
        # Set up action sequences to be optimized. Each of these is a BxNxTx1
        # tensor, where B is the batch size and T is the simulation horizon.
        self.throttle = torch.nn.Parameter(
            torch.zeros(
                batch_size, num_agents, sim_horizon, 1
            )
        )
        # self.throttle.register_hook(lambda grad: print("trottle", grad))
        self.steer = torch.nn.Parameter(
            torch.zeros(
                batch_size, num_agents, sim_horizon, 1
            )
        )
        # we model braking via negative throttle for the adversarial agents
        self.register_buffer(
            "brake_dummy",
            torch.zeros(
                batch_size, num_agents, 1
            )
        )

    def forward(self, observations):
        """
        """
        t = observations["timestep"]

        actions = {
            "throttle": torch.tanh(self.throttle[:, :, t, :] + 1e-3),
            "steer": torch.tanh(self.steer[:, :, t, :]),
            "brake": self.brake_dummy,
        }

        return actions

    def run_step(self, observations):
        """
        For consistency with the driving agents, this provides a "run_step" method
        that simply wraps the forward method.
        """
        return self.forward(observations)

    def initialize_non_critical_actions(self, actions):
        """
        """
        self.original_actions = actions
        with torch.no_grad():
            for batch_id, actions_batch in enumerate(actions):
                for agent_id, action in enumerate(actions_batch):
                    #### throttle
                    num_vars = torch.numel(self.throttle[batch_id][agent_id])
                    tmp = self.throttle[batch_id][agent_id].view(num_vars)
                    for ix in range(min(num_vars, len(action))):
                        tmp[ix] = action[ix]['throttle'].item()

                    ### steer
                    num_vars = torch.numel(self.steer[batch_id][agent_id])
                    tmp = self.steer[batch_id][agent_id].view(num_vars)
                    for ix in range(min(num_vars, len(action))):
                        tmp[ix] = action[ix]['steer']
