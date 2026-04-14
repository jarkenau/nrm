import torch
import pickle

import nrm.dataset.se3 as se3

from nrm.dataset.morphology import sample_morph

from paper_archive.rq3_design_optimisation.ours import ours
from paper_archive.rq3_design_optimisation.baseline import baseline

from paper_archive.utils import bootstrap_mean_ci


device = torch.device("cuda")

pose_error_list_base = []
self_collisions_list_base = []
pose_error_list_ours = []
self_collisions_list_ours = []
for s in range(100):
    torch.manual_seed(s)
    task = se3.random_ball(1000, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])).to(device)
    initial_morph = sample_morph(1, 6, False, device)[0]

    _, pose_error, self_collisions, _, _ = baseline(initial_morph, task, 100)
    pose_error_list_base += [torch.tensor(pose_error)]
    self_collisions_list_base += [torch.tensor(self_collisions)]

    _, pose_error, self_collisions, _, _ = ours(initial_morph, task, 100)
    pose_error_list_ours += [torch.tensor(pose_error)]
    self_collisions_list_ours += [torch.tensor(self_collisions)]

pose_error_base = torch.stack(pose_error_list_base)
self_collisions_base = torch.stack(self_collisions_list_base)

pose_error_ours = torch.stack(pose_error_list_ours)
self_collisions_ours = torch.stack(self_collisions_list_ours)

mean_pose_error_base, lower_pose_error_base, upper_pose_error_base = bootstrap_mean_ci(pose_error_base.numpy())
mean_pose_error_ours, lower_pose_error_ours, upper_pose_error_ours = bootstrap_mean_ci(pose_error_ours.numpy())
mean_self_collisions_base, lower_self_collisions_base, upper_self_collisions_base = bootstrap_mean_ci(self_collisions_base.numpy())
mean_self_collisions_ours, lower_self_collisions_ours, upper_self_collisions_ours = bootstrap_mean_ci(self_collisions_ours.numpy())

pickle.dump([mean_pose_error_base, lower_pose_error_base, upper_pose_error_base], open("data/pose_error_base.pkl", "wb"))
pickle.dump([mean_pose_error_ours, lower_pose_error_ours, upper_pose_error_ours], open("data/pose_error_ours.pkl", "wb"))
pickle.dump([mean_self_collisions_base, lower_self_collisions_base, upper_self_collisions_base], open(
    "data/self_collisions_base.pkl", "wb"))
pickle.dump([mean_self_collisions_ours, lower_self_collisions_ours, upper_self_collisions_ours], open(
    "data/self_collisions_ours.pkl", "wb"))