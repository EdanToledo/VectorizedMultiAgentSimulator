#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Callable, Dict

import torch
from torch import Tensor
import numpy as np
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.collision_reward = kwargs.get("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 1)

        self.plot_grid = True
        self.desired_distance = 0.1
        self.min_collision_distance = 0.005
        self.x_dim = 1
        self.y_dim = 1

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5)
        # Add agents
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            action_script=self.action_script_creator(),
        )
        world.add_agent(self._target)
        goal_entity_filter: Callable[[Entity], bool] = lambda e: not isinstance(
            e, Agent
        )
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.2,
                        entity_filter=goal_entity_filter,
                    )
                ],
                render_action=True,
            )
            agent.collision_rew = np.zeros(batch_dim)
            agent.dist_rew = agent.collision_rew.copy()

            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def action_script_creator(self):
        def action_script(agent, world):
            t = self.t / 30
            agent.action.u = np.stack([np.cos(t), np.sin(t)], axis=1)

        return action_script

    def reset_world_at(self, env_index: int = None):
        target_pos = np.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
           
            dtype=np.float32,
        )

        target_pos[:, Y] = -self.y_dim
        self._target.set_pos(target_pos, batch_index=env_index)
        ScenarioUtils.spawn_entities_randomly(
            self.obstacles + self.world.policy_agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
        )

        for agent in self.world.policy_agents:
            if env_index is None:
                agent.distance_shaping = np.power(
                    np.stack(
                        [
                            np.linalg.norm(
                                agent.state.pos - a.state.pos, axis=-1
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        axis=1,
                    )
                    - self.desired_distance
                , 2).mean(-1) * self.dist_shaping_factor

            else:
                agent.distance_shaping[env_index] = (
                    np.stack(
                        [
                            np.linalg.norm(
                                agent.state.pos[env_index] - a.state.pos[env_index]
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        axis=0,
                    )
                    - self.desired_distance
                ).pow(2).mean(-1) * self.dist_shaping_factor

        if env_index is None:
            self.t = np.zeros(self.world.batch_dim)
        else:
            self.t[env_index] = 0

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0

        if is_first:
            self.t += 1

            # Avoid collisions with each other
            if self.collision_reward != 0:
                for a in self.world.policy_agents:
                    a.collision_rew[:] = 0

                for i, a in enumerate(self.world.agents):
                    for j, b in enumerate(self.world.agents):
                        if j <= i:
                            continue
                        collision = (
                            self.world.get_distance(a, b) <= self.min_collision_distance
                        )
                        if a.action_script is None:
                            a.collision_rew[collision] += self.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += self.collision_reward

        # stay close together (separation)
        agents_dist_shaping = np.power(
            np.stack(
                [
                    np.linalg.norm(agent.state.pos - a.state.pos, axis=-1)
                    for a in self.world.agents
                    if a != agent
                ],
                axis=1,
            )
            - self.desired_distance
        , 2).mean(-1) * self.dist_shaping_factor
        agent.dist_rew = agent.distance_shaping - agents_dist_shaping
        agent.distance_shaping = agents_dist_shaping

        return agent.collision_rew + agent.dist_rew

    def observation(self, agent: Agent):
        return np.concatenate(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self._target.state.pos,
                agent.sensors[0].measure(),
            ],
            axis=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "agent_collision_rew": agent.collision_rew,
            "agent_distance_rew": agent.dist_rew,
        }

        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        # First calculate the closest point to a circle of radius circle_radius given the current position
        circle_origin = np.zeros((1, 2))
        circle_radius = 0.3
        current_pos = observation[:, :2]
        v = current_pos - circle_origin
        closest_point_on_circ = (
            circle_origin + v / np.expand_dims(np.linalg.norm(v, axis=1), 1) * circle_radius
        )

        # calculate the normal vector of the vector from the origin of the circle to that closest point
        # on the circle. Adding this scaled normal vector to the other vector gives us a target point we
        # try to reach, thus resulting in a circular motion.
        closest_point_on_circ_normal = np.stack(
            [closest_point_on_circ[:, Y], -closest_point_on_circ[:, X]], axis=1
        )
        closest_point_on_circ_normal /= np.expand_dims(np.linalg.norm(
            closest_point_on_circ_normal, axis=1
        ), 1)
        closest_point_on_circ_normal *= 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents and obstcles within visibility range
        lidar = observation[:, 6:18]
        object_visible = np.any(lidar < 0.1, axis=1)
        _, object_dir_index = np.min(lidar, axis=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * np.pi
        object_vec = np.stack([np.cos(object_dir), np.sin(object_dir)], axis=1)
        des_pos_object = current_pos - object_vec * 0.1
        des_pos[object_visible] = des_pos_object[object_visible]

        action = np.clip(
            (des_pos - current_pos) * 10,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
