"""Continuous native robot control with scheduled NEW ball feeds.

Only the ball is replaced at each feed boundary. Robot state and WBC raw-action
history persist. This is a practice feeder, not a same-ball rally or an opponent.
Every scored return requires actual table/racket contacts and net clearance.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .runtime import Controller, FORCE_CAPS, stroke_at, sha
from .scoring import ReturnAttempt, TableGeometry
from .interception import PredictiveTimingInterceptor


def relative_feed_state(initial_root, current_root):
    """Aim a NEW feed in the current pelvis XY/yaw frame, using simulator state.

    This is a configurable oracle practice feeder, not robot perception. Height
    remains referenced to the fixed table. No changes occur after ball launch.
    """
    def yaw(pose):
        w,x,y,z=pose[3:7]
        return np.arctan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
    angle=yaw(current_root)-yaw(initial_root)
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s],[s,c]])
    xy=np.asarray(current_root[:2])+rotation@(np.array([-.65,-.25])-initial_root[:2])
    velocity=rotation@np.array([-1.8,0.])
    return np.r_[xy,1.],np.r_[velocity,.5]


def run_continuous(scene, assets, *, feeds=4, interval=3., tail=12., robot_relative_feeder=False,
                   controller_mode="scripted", custom_velocity=None, observation_config=None):
    if not 1 <= feeds <= 32 or interval < 2. or tail < 1.2:
        raise ValueError("Use 1..32 feeds, interval >=2 seconds and tail >=1.2 seconds")
    for duration in (interval, tail):
        if not np.isfinite(duration) or abs(duration/.02-round(duration/.02)) > 1e-8:
            raise ValueError("Durations must be finite multiples of .02 seconds")
    if custom_velocity is not None:
        custom_velocity = np.asarray(custom_velocity, dtype=np.float64)
        if custom_velocity.shape != (3,) or not np.isfinite(custom_velocity).all():
            raise ValueError("custom_velocity must be three finite values")
        if robot_relative_feeder:
            raise ValueError("custom_velocity cannot be combined with robot_relative_feeder")
    if controller_mode != "scripted" and robot_relative_feeder:
        raise ValueError("predictive timing uses a fixed world-frame interception plane; "
                         "it is not supported with robot_relative_feeder")
    if observation_config is not None:
        from .ball_observations import BallObservationStream, ObservationConfig
        from .observed_interception import ObservedTimingInterceptor
        if not isinstance(observation_config, ObservationConfig):
            raise ValueError("observation_config must be an ObservationConfig")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if abs(model.opt.timestep-.00025) > 1e-12 or model.nmocap or model.neq:
        raise ValueError("Native .25 ms free robot and no mocap/equality required")
    controller = Controller(model, assets)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # Once for the entire session.
    mujoco.mj_forward(model, data)
    if observation_config is not None:
        interceptor = ObservedTimingInterceptor(model, controller, mode=controller_mode,
            observation_stream=BallObservationStream(observation_config))
    else:
        interceptor = PredictiveTimingInterceptor(model, controller, mode=controller_mode) if controller_mode != "scripted" else None
    initial_root=data.qpos[:7].copy()
    base_reference = controller.reference(data)
    ball_q = int(model.joint("ball/ball_freejoint").qposadr[0])
    ball_v = int(model.joint("ball/ball_freejoint").dofadr[0])
    ids = {name:model.geom(path).id for name,path in {
        "table":"table/table_top", "net":"table/net", "racket":"robot/racket_collision", "ball":"ball/ball_geom"}.items()}
    center, size = data.geom_xpos[ids["table"]], model.geom_size[ids["table"]]
    geometry = TableGeometry(net_x=float(data.geom_xpos[ids["net"],0]),
        table_x_min=float(center[0]-size[0]),table_x_max=float(center[0]+size[0]),
        table_y_min=float(center[1]-size[1]),table_y_max=float(center[1]+size[1]),
        net_top_z=float(data.geom_xpos[ids["net"],2]+model.geom_size[ids["net"],2]),ball_radius=.02)
    robot_geoms=set()
    for i in range(model.ngeom):
        ancestor=int(model.geom_bodyid[i])
        while ancestor and ancestor!=controller.pelvis:
            ancestor=int(model.body_parentid[ancestor])
        if ancestor==controller.pelvis:robot_geoms.add(i)
    table_geoms={i for i in range(model.ngeom) if model.geom(i).name.startswith("table/")}
    feet={model.body("robot/"+side+"_ankle_roll_link").id for side in ("left","right")}
    known={ids[k]:k for k in ("table","net","racket")}
    nsteps=round(((feeds-1)*interval+tail)/model.opt.timestep)
    feed_stride=round(interval/model.opt.timestep)
    frames=[];reports=[];boundaries=[]
    support_steps=nonfoot_steps=0;cap_ratio=0.;fallen_any=False;failure=None
    scorer=None;feed_id=-1;feed_started=0.
    for step in range(nsteps):
        if step%feed_stride==0 and step//feed_stride<feeds:
            if scorer is not None:
                reports.append({"feed_id":feed_id,"launched_at":feed_started,"ball_outcome":scorer.finish()})
            feed_id+=1;feed_started=step*model.opt.timestep
            before=data.qpos[:ball_q].copy();history_before=controller.previous.copy()
            if robot_relative_feeder:
                position, velocity = relative_feed_state(initial_root, data.qpos[:7])
            elif custom_velocity is not None:
                position = np.array([-.65, -.25, 1.])
                velocity = custom_velocity
            else:
                position, velocity = np.array([-.65, -.25, 1.]), np.array([-1.8, 0., .5])
            data.qpos[ball_q:ball_q+7]=np.r_[position,1.,0.,0.,0.]
            data.qvel[ball_v:ball_v+6]=np.r_[velocity,0.,0.,0.]
            data.qacc_warmstart[ball_v:ball_v+6]=0.
            mujoco.mj_forward(model,data)
            if interceptor is not None:
                interceptor.reset_feed(feed_id, data)
            boundaries.append({"feed_id":feed_id,"time":feed_started,
                "launch_position":position.tolist(),"launch_velocity":velocity.tolist(),
                "robot_qpos_change_at_feed":float(np.max(abs(data.qpos[:ball_q]-before))),
                "wbc_history_change_at_feed":float(np.max(abs(controller.previous-history_before)))})
            scorer=ReturnAttempt(geometry)
        if step%80==0:
            relative=max(0.,data.time-feed_started)
            ref=base_reference.copy()
            raw7=interceptor.step(data, relative) if interceptor is not None else stroke_at(relative)
            ref[10+controller.arm]+=.7*raw7
            ref[10:]=np.clip(ref[10:],controller.ranges[:,0],controller.ranges[:,1])
            raw29,obs=controller.apply(data,ref)
        mujoco.mj_step(model,data)
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or abs(data.time-(step+1)*model.opt.timestep)>1e-7
            or any(data.warning[k].number for k in (mujoco.mjtWarning.mjWARN_BADQPOS,
                   mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC))):
            failure="physics_warning_or_reset";break
        active=set();support=nonfoot=False
        for contact in data.contact[:data.ncon]:
            if contact.efc_address<0 or contact.dist>0:continue
            a,b=int(contact.geom1),int(contact.geom2)
            if (a in robot_geoms and b in table_geoms) or (b in robot_geoms and a in table_geoms):support=True
            for ground,other in ((a,b),(b,a)):
                if model.geom_type[ground]==mujoco.mjtGeom.mjGEOM_PLANE and other in robot_geoms and int(model.geom_bodyid[other]) not in feet:nonfoot=True
            if ids["ball"] in (a,b):
                other=b if a==ids["ball"] else a
                active.add(known.get(other,"floor" if model.geom_type[other]==mujoco.mjtGeom.mjGEOM_PLANE else "body"))
        support_steps+=int(support);nonfoot_steps+=int(nonfoot)
        quat=data.qpos[3:7]
        fallen=data.qpos[2]<.45 or -(1.-2.*(quat[1]**2+quat[2]**2))>-.5
        fallen_any|=bool(fallen)
        scorer.observe(time=data.time,ball_position=data.qpos[ball_q:ball_q+3],contacts=active,fallen=fallen)
        cap_ratio=max(cap_ratio,float(np.max(abs(data.actuator_force[controller.actuator_ids])/FORCE_CAPS)))
        if (step+1)%80==0:
            frames.append({"time":float(data.time),"trial_time":float(data.time-feed_started),"trial_id":feed_id,
                "qpos":data.qpos.copy(),"qvel":data.qvel.copy(),"ctrl":data.ctrl.copy(),
                "raw7":raw7.copy(),"reference39":ref.copy(),"raw_wbc29":raw29.copy(),"observation132":obs.copy()})
    reports.append({"feed_id":feed_id,"launched_at":feed_started,"ball_outcome":scorer.finish(aborted=failure is not None)})
    accepted=bool(not fallen_any and not support_steps and not nonfoot_steps and cap_ratio<=1.000001 and failure is None and step+1==nsteps)
    report={"protocol":"g1-ping-pong-lab.native-cpu-continuous-feeder-v1",
        "description":"One continuously simulated robot. Only the ball is replaced at scheduled feed boundaries. These are separate fixed feeds, not a continuous same-ball rally.",
        "controller_mode":controller_mode,
        "feeds":reports,"feed_boundaries":boundaries,"feed_interval_seconds":interval,
        "feeder_frame":"current simulator pelvis XY/yaw (oracle pose)" if robot_relative_feeder else "fixed world frame",
        "requested_seconds":(feeds-1)*interval+tail,"actual_seconds":float(data.time),"physics_substeps":step+1,
        "robot_resets_after_start":0,"wbc_history_resets_after_start":0,"ball_replacements_after_start":len(boundaries)-1,
        "fallen":fallen_any,"external_support_substeps":support_steps,"nonfoot_ground_substeps":nonfoot_steps,
        "maximum_force_cap_ratio":cap_ratio,"failure":failure,"full_session_stance_accepted":accepted,
        "accepted_returns":sum(bool(r['ball_outcome']['success']) for r in reports) if accepted else 0}
    if observation_config is not None:
        report["ball_observations"] = interceptor.observation_stream.summary()
        report["observation_trace"] = interceptor.observation_stream.trace
    return report,frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=Path('assets/scene.xml'))
    parser.add_argument('--wbc',type=Path,default=Path('assets/wbc'))
    parser.add_argument('--feeds',type=int,default=4)
    parser.add_argument('--interval',type=float,default=3.)
    parser.add_argument('--tail',type=float,default=12.)
    parser.add_argument('--output',type=Path,default=Path('outputs/continuous-feeds'))
    parser.add_argument('--robot-relative-feeder',action='store_true',help='Aim new feeds in current simulator pelvis XY/yaw frame; no postlaunch edits')
    parser.add_argument('--controller',choices=['scripted','predictive-timing'],default='scripted',help='Arm trajectory controller strategy (default: scripted)')
    parser.add_argument('--velocity',nargs=3,type=float,default=None,help='Custom feed velocity [vx, vy, vz] for perturbation testing')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report,frames=run_continuous(args.scene,args.wbc,feeds=args.feeds,interval=args.interval,tail=args.tail,
                               robot_relative_feeder=args.robot_relative_feeder,
                               controller_mode=args.controller,custom_velocity=args.velocity)
    np.savez_compressed(args.output/'session.npz',**{k:np.array([f[k]for f in frames])for k in frames[0]})
    report.update(scene_sha256=sha(args.scene),source_sha256=sha(__file__),trajectory_sha256=sha(args.output/'session.npz'))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['accepted_returns','full_session_stance_accepted','fallen','actual_seconds']}
                     |{'feed_results':[r['ball_outcome']['reason'] for r in report['feeds']]}),flush=True)


if __name__=='__main__':main()
