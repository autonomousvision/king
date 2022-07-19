import torch
from torch.nn import functional as F

# Builds on the implentation from "World on Rails", see king/external_code/wor.


class BicycleModel(torch.nn.Module):
    """
    """
    def __init__(self, delta_t):
        super().__init__()
        self.delta_t = delta_t

        # values taken from "World On Rails"
        self.register_buffer("front_wb", torch.tensor([-0.090769015]))
        self.register_buffer("rear_wb", torch.tensor([1.4178275]))
        self.register_buffer("steer_gain", torch.tensor([0.36848336]))
        self.register_buffer("brake_accel", torch.tensor([-4.952399]))
        self.register_buffer("throt_accel", torch.tensor([[0.5633837]]))

    def forward(self, state, actions, is_terminated):
        """
        Computes the next from the current state given associated actions using
        the bicycle model tuned for CARLA from the "World On Rails" paper

        https://arxiv.org/abs/2105.00636

        Args:
            -
        Returns:
            -
        """
        braking_ego = actions["brake"]
        braking_adv = torch.lt(actions["throttle"], torch.zeros_like(actions["throttle"]))
        accel = braking_ego * self.brake_accel + \
            braking_adv * -self.brake_accel * actions["throttle"] + \
            ~braking_adv * self.throt_accel * actions["throttle"]

        wheel = self.steer_gain * actions["steer"]
        beta = torch.atan(
            self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel)
        )

        speed = torch.norm(state["vel"], dim=-1, keepdim=True)
        motion_components = torch.cat(
            [torch.cos(state["yaw"]+beta), torch.sin(state["yaw"]+beta)],
            dim=-1,
        )

        update_mask = ~is_terminated.view(-1, 1, 1)
        state["pos"] = state["pos"] + speed * motion_components * self.delta_t * update_mask
        state["yaw"] = state["yaw"] + speed / self.rear_wb * torch.sin(beta) * self.delta_t * update_mask

        speed = F.softplus(speed + accel * self.delta_t, beta=7)
        state["vel"] = state["vel"] + (speed * motion_components - state["vel"]) * update_mask

        return state
