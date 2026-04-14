import torch.utils.dlpack

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

import jax
import jax.numpy as jnp
import optax
import jax.dlpack

import nrm.dataset.se3 as se3

from nrm.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from nrm.dataset.self_collision import collision_check
from nrm.dataset.self_collision import LINK_RADIUS, EPS
from paper_archive.utils import jax_distance, jax_forward_kinematics, jax_inverse_kinematics


@jax.custom_vjp
def squasher(param):
    mask = (jnp.abs(param) >= 2 * LINK_RADIUS).astype(param.dtype)
    return param * mask


def squasher_fwd(param):
    return squasher(param), None


def squasher_bwd(res, g):
    # Straight-Through Estimator: Pass gradient identically
    return (g,)


squasher.defvjp(squasher_fwd, squasher_bwd)


@jax.custom_vjp
def normaliser(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return param / safe_norm


def normaliser_fwd(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return (param / safe_norm), (param, l2_norm, safe_norm)


def normaliser_bwd(res, g):
    param, l2_norm, safe_norm = res
    safe_l2_norm = jnp.maximum(l2_norm, 1e-12)

    chain = jnp.where(
        jnp.any(jnp.abs(param) > EPS, axis=1, keepdims=True),
        param / safe_l2_norm,
        jnp.zeros_like(param)
    )
    grad_param = (g * safe_norm - chain * jnp.sum(g * param)) / (safe_norm ** 2)
    return (grad_param,)


normaliser.defvjp(normaliser_fwd, normaliser_bwd)


def preprocess(param):
    norm_param = normaliser(param)
    squashed = squasher(norm_param)
    return normaliser(squashed), norm_param


def loss_fn(lengths, alpha, task_poses, init_joints):
    param, norm_lengths = preprocess(lengths)

    morph = jnp.concatenate([alpha, param], axis=1)
    bmorph = jnp.broadcast_to(morph, (task_poses.shape[0], *morph.shape))
    optimal_joints = jax_inverse_kinematics(morph, task_poses, init_joints)

    reached_poses = jax.vmap(jax_forward_kinematics)(bmorph, optimal_joints)

    ee_poses = reached_poses[:, -1, :, :]
    dists = jax.vmap(jax_distance)(ee_poses, task_poses)

    loss = jnp.mean(dists)

    return loss, (morph,)


loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


def baseline(initial_morph: Float[Tensor, "dofp1 3"], task: Float[Tensor, "num_samples 4 4"], n_iter: int,
             logging: bool = True) \
        -> tuple[
            list[float],  # Train Loss
            list[float],  # Pose Error
            list[float],  # Self-Collisions
            list[Tensor],  # Morphs
            dict[str, list[float]]  # debug
        ]:
    task_torch = task.clone()
    task = jax.dlpack.from_dlpack(task.contiguous().clone())
    alpha = jax.dlpack.from_dlpack(initial_morph[:, 0:1].contiguous().clone())
    lengths = jax.dlpack.from_dlpack(initial_morph[:, 1:].contiguous().clone())

    train_loss = []
    pose_error = []
    self_collisions = []
    morphs = []

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.01)
    )
    opt_state = optimizer.init(lengths)

    for i in tqdm(range(n_iter)):
        key = jax.random.PRNGKey(i)
        init_joints_jax = jax.random.uniform(key, shape=(task.shape[0], alpha.shape[0], 1), minval=-jnp.pi,
                                             maxval=jnp.pi)

        (loss, (morph,)), grads = loss_and_grad_fn(lengths, alpha, task, init_joints_jax)

        updates, opt_state = optimizer.update(grads, opt_state, lengths)
        lengths = optax.apply_updates(lengths, updates)

        # Logging
        if logging:
            train_loss += [loss.item()]
            morph_torch = torch.from_dlpack(morph)
            bmorph = morph_torch.unsqueeze(0).expand(task.shape[0], -1, -1)
            joints = numerical_inverse_kinematics(morph_torch, task_torch)[0]
            reached_pose = forward_kinematics(bmorph, joints)
            pose_error += [se3.distance(reached_pose[:, -1, :, :], task_torch).squeeze(-1).mean().item()]
            self_collisions += [collision_check(bmorph, reached_pose).sum().item()]
            morphs += [morph_torch.clone().cpu()]

    return train_loss, pose_error, self_collisions, morphs, {}
