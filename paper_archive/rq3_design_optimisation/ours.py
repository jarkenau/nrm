import torch

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

import nrm.dataset.se3 as se3

from nrm.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from nrm.dataset.self_collision import collision_check
from nrm.dataset.self_collision import LINK_RADIUS, EPS
from nrm.model import MLP


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        # Keep the hard mask for the forward pass
        mask = (param.abs() >= 2 * LINK_RADIUS).float()
        return param * mask

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: Pass the gradient exactly as is
        # This prevents the gradient from vanishing when the parameter is 0
        return grad_output


class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output):
        param, l2_norm, norm = ctx.saved_tensors

        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param)
        )

        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def preprocess(param):
    norm_param = Normaliser.apply(param)
    squashed = SquasherSTE.apply(norm_param)
    return Normaliser.apply(squashed), norm_param


def ours(initial_morph: Float[Tensor, "dofp1 3"], task: Float[Tensor, "num_samples 4 4"], n_iter: int,
         logging:bool=True) \
        -> tuple[
            list[float],  # Train Loss
            list[float],  # Pose Error
            list[float],  # Self-Collisions
            list[Tensor],  # Morphs
            dict[str, list[float]]  # debug
        ]:
    task_vec = se3.to_vector(task)
    alpha = initial_morph[:, 0:1].clone()

    lengths = initial_morph[:, 1:]
    lengths.requires_grad = True

    train_loss = []
    pose_error = []
    self_collisions = []
    morphs = []
    predicted_reachability = []

    optimizer = torch.optim.AdamW([lengths], lr=0.01)
    model = MLP.from_id(13).to(initial_morph.device)
    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()

        param, norm_lengths = preprocess(lengths)

        morph = torch.cat([alpha, param], dim=1)
        bmorph = morph.unsqueeze(0).expand(task.shape[0], -1, -1)
        logit = model(bmorph, task_vec)

        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, torch.ones_like(logit))
        # + (torch.nn.Softplus(5)(2 * LINK_RADIUS - norm_lengths.abs())).sum()

        loss.backward()
        optimizer.step()

        # Logging
        if logging:
            with torch.no_grad():
                train_loss += [loss.item()]
                joints = numerical_inverse_kinematics(morph, task)[0]
                reached_pose = forward_kinematics(bmorph, joints)
                pose_error += [se3.distance(reached_pose[:, -1, :, :], task).squeeze(-1).mean().item()]
                self_collisions += [collision_check(bmorph, reached_pose).sum().item()]
                morphs += [morph.detach().clone().cpu()]
                predicted_reachability += [torch.nn.Sigmoid()(logit).mean().item()]
    return train_loss, pose_error, self_collisions, morphs, {"Predicted Reachability": predicted_reachability}
