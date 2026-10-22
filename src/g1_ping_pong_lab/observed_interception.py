"""Observation adapter around Pernav's unchanged timing planner.

This isolates sensing experiments from predictor/filter development. No
measurements are available during delay startup; the arm holds READY then.
"""
import numpy as np

from .interception.controller import PredictiveTimingInterceptor
from .runtime import stroke_at


class ObservedTimingInterceptor(PredictiveTimingInterceptor):
    def __init__(self, model, controller, *, observation_stream, mode="predictive-timing"):
        super().__init__(model, controller, mode=mode)
        self.observation_stream = observation_stream

    def reset_feed(self, feed_id, data):
        # Deliberately do not call the parent's oracle-state initialization.
        self.feed_id = feed_id
        self.current_plan = None
        self.observation_stream.reset_feed(feed_id)

    def step(self, data, feed_elapsed):
        stream = self.observation_stream
        sample = stream.observe(feed_elapsed, data.qpos[self.ball_q:self.ball_q+3],
                                data.qvel[self.ball_v:self.ball_v+3])
        if self.mode == "scripted":
            stream.record_plan(feed_elapsed)
            return stroke_at(feed_elapsed)
        if sample is None:
            stream.record_plan(0.)
            return np.zeros(7)
        can_replan = (sample.velocity[0] < -.1 and sample.position[0] > -1.25
                      and feed_elapsed < .24)
        replanned = self.current_plan is None or can_replan
        if replanned:
            # Uncompensated deliberately treats delivered data as current.
            # Timestamp-aware predicts from capture time, without future states.
            clock = sample.captured_at if stream.config.timestamp_aware else feed_elapsed
            self.current_plan = self.planner.plan(sample.position, sample.velocity,
                                                  feed_elapsed=clock, target_x=self.target_x)
        tau = self.planner.effective_stroke_time(feed_elapsed, self.current_plan)
        stream.record_plan(tau, self.current_plan.target_time, replanned=replanned)
        return stroke_at(tau)
