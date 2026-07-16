from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.unwrapped.get_action_mask()

def wrap_with_mask(env):
    return ActionMasker(env, mask_fn)