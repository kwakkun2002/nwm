import torch
from evo.core import metrics, sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe


def actions_to_traj(actions):
    positions_xyz = torch.zeros((actions.shape[0], 3))
    positions_xyz[:, :2] = actions
    orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))
    orientations_quat_wxyz[:, -1] = 1
    timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
    return PoseTrajectory3D(
        positions_xyz=positions_xyz,
        orientations_quat_wxyz=orientations_quat_wxyz,
        timestamps=timestamps,
    )


def eval_metrics(traj_ref, traj_pred):
    traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

    result = main_ape.ape(
        traj_ref,
        traj_pred,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=False,
        correct_scale=False,
    )
    ate = result.stats["rmse"]

    result = main_rpe.rpe(
        traj_ref,
        traj_pred,
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=False,
        correct_scale=False,
        delta=1.0,
        delta_unit=metrics.Unit.frames,
        rel_delta_tol=0.1,
    )
    rpe_rot = result.stats["rmse"]

    result = main_rpe.rpe(
        traj_ref,
        traj_pred,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=False,
        correct_scale=False,
        delta=1.0,
        delta_unit=metrics.Unit.frames,
        rel_delta_tol=0.1,
    )
    rpe_trans = result.stats["rmse"]

    return ate, rpe_trans, rpe_rot
