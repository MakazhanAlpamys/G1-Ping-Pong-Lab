# Longer-session probes

These native reports show the limits observed while preparing the demo:

| Feeder | Feeds | Legal returns | Audited duration |
| --- | ---: | ---: | ---: |
| Fixed world frame, 3 s interval | 4 | 2 | 21 s |
| Fixed world frame, 4 s interval | 4 | 2 | 24 s |
| Current simulated robot position/yaw, 3 s interval | 4 | 1 | 21 s |

The robot and balance-controller history stayed continuous in every probe.
Only the ball was replaced at scheduled feed boundaries. All three full stance
audits passed. A stable robot does not imply a correct return.

The first two reports predate addition of the optional robot-relative feeder
flag; their source hashes identify that historical development revision. The
current runtime keeps fixed world-frame feeds as its default. The released
two-feed recording has its own current-source report and verification receipt
in the parent directory. These reports establish the observed limits, rather
than an estimate of success against unknown feeds.

## Separate ball-observation experiment

The [noise and latency benchmark](../noise-latency.md) and its
[32-trial pilot evidence](noise-latency-pilot.json) use independent reset trials
with synthetic sensing errors. They are separate from the continuous endurance
probes above and do not establish longer rallies.
