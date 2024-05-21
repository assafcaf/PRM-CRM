from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv


def get_env_creator(
    env,
    num_agents,
    ep_length=1000,
    metric=0,
    same_color=False,
    gray_scale=False,
    same_dim=same_dim
):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=False,
                ep_length=ep_length,
                metric=metric,
                same_color=same_color,
                gray_scale=gray_scale,
                same_dim=same_dim
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,

            )

    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")

    return env_creator
