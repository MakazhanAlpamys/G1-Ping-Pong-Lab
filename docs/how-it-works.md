# How it works

The demonstration combines an existing balance policy with a small, optimized
arm motion. It does not train a new general table-tennis policy.

## Control loop

1. Initialize the robot, ball and controller history at the start of the session.
2. Initialize a controlled incoming ball at each scheduled feed boundary.
3. At 50 Hz, add a seven-joint right-arm stroke to the standing reference.
4. Project the reference into the native joint limits.
5. Build the pretrained controller's 132-value observation and run ONNX inference.
6. Convert its 29 outputs into PD joint targets with the original gains and force caps.
7. Advance MuJoCo at a 0.25 ms physics timestep.

The arm script contains a preparation pose, a forward pose, a start time and a
duration. Cross-entropy method (CEM) search was used to select these parameters
in earlier simulation experiments. This release freezes the chosen stroke and
provides the runtime, rather than the larger experimental training workspace.

The controller observes simulated robot state and a reference motion. The stroke
timing is fixed for the known feed; it does not use camera images or adapt to a
human opponent. Equal input/output dimensions do not make an arbitrary replacement
policy compatible with this runtime.

## What counts as a return

The scorer follows actual ball contact events: an incoming bounce on the robot's
half, racket contact, a legal net crossing and a bounce on the opposite half.
Geometric proximity, reward values and visually plausible swings do not count.

Acceptance also requires the complete native observation window: 15 seconds for
the continuous demo, or 12 seconds per independent baseline trial. There must be no fall, no table
or net support, no non-foot ground contact and no exceeded native actuator caps.
For the main demo, the ball is replaced at the next scheduled feed boundary;
its path is never changed between launch and the following boundary. The robot
and controller history continue uninterrupted. The reference two-feed session
launches at 0 and 3 seconds and runs to 15 seconds, auditing stance throughout.

A separate repeatability protocol resets the full robot and controller history
between independent 12-second trials. Its eight successful repetitions are a
different result from the two consecutive returns in the original demo.

## Rendering

The simulator records generalized positions, velocities, commands and trial IDs.
The renderer selects actual saved states; it never interpolates a ball strike or
reintegrates the physics. Studio lighting, table legs, boundary lines and cameras
are presentation changes only. They do not add collision surfaces or change the
recorded movement. The ball keeps its physical 40 mm diameter.

The [published original demo](../media/demo.mp4) shows the first six seconds of
one session, at normal speed and with a single camera path, without montage cuts.
Its [render receipt](../media/demo.render.json) records the exact source-frame
mapping. Validation uses the full 15-second physical session, including the
subsequent balance observation, rather than the shorter video. That complete
recording is included as `data/continuous-session.npz`; a full-length video can
be generated locally.

## Toward a real robot

This release has no real-hardware deployment path. Such work would require ball
tracking, calibrated table and robot coordinates, latency-aware prediction,
hardware system identification and a validated safe controller. The simulated
state available here does not establish that perception or sim-to-real transfer works.
