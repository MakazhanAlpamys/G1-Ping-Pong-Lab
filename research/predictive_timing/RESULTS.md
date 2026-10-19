# Empirical Benchmark Results: Predictive Impact Timing

**Repository:** `MakazhanAlpamys/G1-Ping-Pong-Lab`  
**Feature Branch:** `feat/predictive-impact-timing`  
**Base Commit:** `95dd0bd2c4e49e4a0049b1b8e3c81cd8de0696a3`  
**Date:** October 2026  
**Test Environment:** Apple Silicon (macOS ARM64, Python 3.12.13, MuJoCo 3.11.0, ONNX Runtime 1.31.0)  
**Stored Artifacts:**
- JSON Summary: `outputs/predictive_timing/benchmark-summary.json`
- Tabular CSV: `outputs/predictive_timing/benchmark-results.csv`

---

## 1. Executive Summary

This document reports the empirical validation of the isolated predictive ball-flight timing controller on branch `feat/predictive-impact-timing`.

### Summary of Observed Performance:
1. **Faster Incoming Feeds (+3% to +12%):**
   - In the specific faster-feed evaluations conducted (+3%, +5%, +7%, +12%), predictive timing achieved a valid return on each tested condition (4/4).
   - Under the same conditions, the scripted baseline achieved 0/4 valid returns, failing via net contact (+3%), body collision (+5%), out of bounds landing (+7%), and table drop (+12%).
2. **Speed Boundary (+15%):**
   - At +15% launch speed ($v_x = -2.07\text{ m/s}$), both controllers failed. The baseline struck the ball late and dropped it to the floor (`floor_contact`), while predictive timing initiated swing earlier but the ball clipped the robot's pelvis during backswing (`body_contact`).
3. **Multi-Feed Endurance Invariance:**
   - In 4-feed continuous endurance testing, both `scripted` and `predictive-timing` achieved 2/4 valid returns and 3/4 contacts.
   - In 8-feed continuous endurance testing, both controllers achieved 2/8 valid returns and 3/8 contacts.
   - Analysis confirms that endurance failure is caused by floating-base stance drift ($> 3.5\text{ cm}$ rearward along the X axis), not arrival timing.
4. **Baseline Bit-for-Bit Identity:**
   - Running the default CLI without arguments generates trajectory data (`session.npz`) whose SHA256 checksum matches the upstream reference commit `95dd0bd` bit-for-bit.

---

## 2. Physical Model, Calibration, and Validation

### 2.1 Trajectory Equations and Parameters
The analytical predictor models ball flight in two piecewise ballistic phases separated by a table bounce:

1. **Pre-Bounce Ballistics:**
   Free flight under gravity $g = 9.81\text{ m/s}^2$ until the ball sphere center reaches the table contact elevation:
   $$z_{\text{contact}} = z_{\text{table}} + r_{\text{ball}} = 0.760\text{ m} + 0.020\text{ m} = 0.780\text{ m}$$
   Time to bounce:
   $$\Delta t_b = \frac{v_z + \sqrt{v_z^2 + 2g(z - z_{\text{contact}})}}{g}$$

2. **Table Bounce Model:**
   In MuJoCo, the table top geom has soft constraint parameters `solref=(0.005, 0.134)` and friction `0.1`. The macroscopic post-bounce velocity is modeled by two calibrated restitution parameters:
   $$\mathbf{v}_{\text{post}} = \begin{bmatrix} k_x \cdot v_{x, \text{pre}} \\ k_x \cdot v_{y, \text{pre}} \\ -e_z \cdot v_{z, \text{pre}} \end{bmatrix}$$
   - $e_z = 0.65895$: **Normal coefficient of restitution**, calibrated to match post-bounce vertical apex in MuJoCo.
   - $k_x = 0.8520$: **Tangential velocity retention ratio** ($v_{x, \text{post}} / v_{x, \text{pre}}$). Note: $k_x$ is an empirical velocity ratio, *not* a Coulomb friction coefficient.
   - **Applicable Conditions:** Flat horizontal table collisions with negligible spin.

3. **Post-Bounce Propagation:**
   Ballistic flight from bounce point $(x_b, y_b, z_{\text{contact}})$ to the nominal interception plane $x_{\text{target}} = -1.47158\text{ m}$:
   $$\Delta t_{\text{post}} = \frac{x_{\text{target}} - x_b}{v_{x, \text{post}}}$$
   $$t^* = t_b + \Delta t_{\text{post}}$$

### 2.2 Prediction Accuracy Audit Against MuJoCo Ground Truth
Ground truth was measured by running MuJoCo at 4000 Hz ($\Delta t = 0.25\text{ ms}$) on the nominal feed ($v_0 = [-1.8, 0, 0.5]\text{ m/s}$) and performing linear interpolation across the substep crossing $x = -1.47158\text{ m}$:

| Event / Parameter | Actual MuJoCo Event | Analytical Prediction | Error |
| :--- | :---: | :---: | :---: |
| **Table Bounce Time $t_b$** | $0.2690\text{ s}$ | $0.2690\text{ s}$ | $< 0.25\text{ ms}$ (within 1 substep) |
| **Table Bounce Position $x_b$** | $-1.1342\text{ m}$ | $-1.1342\text{ m}$ | $< 1.0\text{ mm}$ |
| **Interception Time $t^*$ ($x=-1.47158$)** | $0.4889\text{ s}$ | $0.4890\text{ s}$ | $+0.11\text{ ms}$ |
| **Interception Height $z^*$ ($x=-1.47158$)** | $0.8535\text{ m}$ | $0.8522\text{ m}$ | $-1.24\text{ mm}$ |
| **Execution Latency** | — | $< 5\ \mu\text{s}$ | — |

The timing error ($+0.11\text{ ms}$) is smaller than a single MuJoCo physics step ($0.25\text{ ms}$), providing suitable accuracy for 50 Hz control updates ($\Delta t_{\text{ctrl}} = 20\text{ ms}$).

---

## 3. Suite 1: Continuous Multi-Feed Endurance Benchmark

Simulated across 4-feed ($15.0\text{ s}$) and 8-feed ($33.0\text{ s}$) continuous sessions under 4000 Hz physics, 50 Hz WBC, $3.0\text{ s}$ feed intervals, and $12.0\text{ s}$ tail.

### 3.1 Four-Feed Endurance Summary

| Controller | Valid Returns | Contacts | Stance Balance | Feed-by-Feed Outcomes |
| :--- | :---: | :---: | :---: | :--- |
| **`scripted` (Baseline)** | 2 / 4 ($50.0\%$) | 3 / 4 ($75.0\%$) | **Accepted** | `valid_return`, `valid_return`, `floor_contact`, `floor_contact` |
| **`predictive-timing`** | 2 / 4 ($50.0\%$) | 3 / 4 ($75.0\%$) | **Accepted** | `valid_return`, `valid_return`, `floor_contact`, `floor_contact` |

### 3.2 Eight-Feed Endurance Summary

| Controller | Valid Returns | Contacts | Stance Balance | Max Force Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **`scripted` (Baseline)** | 2 / 8 ($25.0\%$) | 3 / 8 ($37.5\%$) | **Accepted** | 1.0000 |
| **`predictive-timing`** | 2 / 8 ($25.0\%$) | 3 / 8 ($37.5\%$) | **Accepted** | 1.0000 |

*Endurance Analysis:*
Multi-feed outcomes are identical between controllers. Over successive strokes, reactive balance impulses cause the floating base to translate rearward ($> 3.5\text{ cm}$ by Feed 2). Because feeds are delivered to fixed world coordinates, the ball reaches the displaced robot at the same late physical position. Predictive timing alone does not resolve stance translation.

---

## 4. Suite 2: Feed Velocity Perturbation Sweep

Launch velocity $v_x$ was evaluated across seven discrete values from $-1.53\text{ m/s}$ ($-15\%$) to $-2.07\text{ m/s}$ ($+15\%$) around the nominal $-1.80\text{ m/s}$ feed.

| Launch $v_x$ | Delta | Scripted Baseline | Predictive Timing | Contact Details (Predictive Timing) |
| :---: | :---: | :--- | :--- | :--- |
| **$-1.53\text{ m/s}$** | $-15\%$ | `floor_contact` (No contact) | `floor_contact` (No contact) | Ball drops short of arm reach |
| **$-1.62\text{ m/s}$** | $-10\%$ | `floor_contact` (No contact) | `floor_contact` (No contact) | Ball drops short of arm reach |
| **$-1.71\text{ m/s}$** | $-5\%$ | `floor_contact` (No contact) | `floor_contact` (No contact) | Ball drops short of arm reach |
| **$-1.80\text{ m/s}$** | **$0\%$** | **`valid_return` (Success)** | **`valid_return` (Success)** | Impact $t=0.489\text{ s}$, pos $(-1.4716, -0.2500, 0.8535)$ |
| **$-1.89\text{ m/s}$** | **$+5\%$** | `body_contact` (Fail) | **`valid_return` (Success)** | Impact $t=0.464\text{ s}$, pos $(-1.4720, -0.2500, 0.8691)$ |
| **$-1.98\text{ m/s}$** | **$+10\%$** | `wrong_first_landing` (Fail) | **`valid_return` (Success)** | Impact $t=0.440\text{ s}$, pos $(-1.4719, -0.2500, 0.8778)$ |
| **$-2.07\text{ m/s}$** | **$+15\%$** | `floor_contact` (Fail) | `body_contact` (Fail) | Impact $t=0.419\text{ s}$, clips pelvis |

*Findings:*
- At $+5\%$ and $+10\%$ speed, predictive timing advances the stroke clock ($\tau = t - (t^* - 0.489)$), impacting the ball cleanly at $x = -1.472\text{ m}$ and returning it across the net.
- Slower feeds ($-15\%$ to $-5\%$) fail for both controllers because the ball bounces short and drops below the racket envelope.
- At $+15\%$, the ball arrives too quickly for the backswing to clear, resulting in pelvis contact.

---

## 5. Suite 3: Repeatability and Intermediate Variations

Six specific variation test cases were executed to inspect intermediate speed increments and lateral deviations:

| Case | Launch Velocity $(v_x, v_y, v_z)$ | Speed Delta | Scripted Baseline | Predictive Timing |
| :--- | :---: | :---: | :--- | :--- |
| **`nominal`** | $[-1.800, 0.0, 0.5]$ | $0.0\%$ | **`valid_return`** | **`valid_return`** |
| **`plus_3pct`** | $[-1.854, 0.0, 0.5]$ | $+3.0\%$ | `net_contact` | **`valid_return`** |
| **`plus_7pct`** | $[-1.926, 0.0, 0.5]$ | $+7.0\%$ | `wrong_first_landing` | **`valid_return`** |
| **`plus_12pct`**| $[-2.016, 0.0, 0.5]$ | $+12.0\%$ | `floor_contact` | **`valid_return`** |
| **`lateral_left`** | $[-1.800, -0.05, 0.5]$ | Lateral $-0.05\text{ m/s}$ | `net_contact` | `net_contact` |
| **`lateral_right`**| $[-1.800, +0.05, 0.5]$ | Lateral $+0.05\text{ m/s}$ | `net_contact` | `net_contact` |

*Trial Count & Results:*
- Total evaluated variation trials: **12 executions** (6 test conditions $\times$ 2 controllers).
- In the 4 tested positive longitudinal perturbations (+3%, +5%, +7%, +12%), predictive timing achieved valid returns in all 4 trials, compared to 0/4 for the baseline.
- In the 2 lateral deviation trials ($\pm 0.05\text{ m/s}$ $v_y$), both controllers failed with `net_contact`, indicating that lateral redirection requires adjusting racket yaw rather than purely longitudinal timing.

---

## 6. Baseline Bit-for-Bit Compatibility

A bit-level comparison was performed between the upstream repository state and the modified default CLI:

- **Reference Commit:** `95dd0bd2c4e49e4a0049b1b8e3c81cd8de0696a3` (`origin/main`)
- **Reference Command:** `python -m g1_ping_pong_lab.continuous_feeds --feeds 4 --interval 3 --tail 12 --output outputs/baseline_ref`
- **Modified CLI Command:** `python -m g1_ping_pong_lab.continuous_feeds --feeds 4 --interval 3 --tail 12 --output outputs/baseline_test`
- **Trajectory Checksum Comparison:**
  - `outputs/baseline_ref/session.npz`: `9ea754a03165765ffdc5f4ea95541d4eeb5ea22dd7f44f67368dafdf4a366a6f`
  - `outputs/baseline_test/session.npz`: `9ea754a03165765ffdc5f4ea95541d4eeb5ea22dd7f44f67368dafdf4a366a6f`
- **Result:** The trajectory checksums are bit-for-bit identical. Default execution remains bit-exact reproducible with the original baseline.
