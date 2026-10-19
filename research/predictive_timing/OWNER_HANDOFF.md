# Contribution Handoff: Predictive Ball-Flight Impact Timing

**Target Repository:** https://github.com/MakazhanAlpamys/G1-Ping-Pong-Lab  
**Feature Branch:** `feat/predictive-impact-timing`  
**Base Commit SHA:** `95dd0bd2c4e49e4a0049b1b8e3c81cd8de0696a3`  
**Author:** Pernav  
**Status:** Audited, verified, ready to push to contributor fork  

---

## 1. Scope of Contribution

This contribution provides an analytical ball-flight and impact-timing predictor for the open-loop arm trajectory pipeline in `G1-Ping-Pong-Lab`.

### The Baseline Limitation:
The original arm trajectory (`stroke_at(t)`) executes at fixed, predetermined time intervals ($t_{\text{start}} = 0.2633\text{ s}$, duration $0.1717\text{ s}$). While calibrated for a single nominal serve velocity ($-1.80\text{ m/s}$), it fails when incoming feeds arrive faster because the arm stroke executes too late relative to the approaching ball.

### What This PR Implements:
1. **Piecewise Analytical Predictor (`interception/predictor.py`):**
   - Calculates free-flight ballistics under gravity ($g = 9.81\text{ m/s}^2$).
   - Models horizontal table contact using calibrated normal restitution ($e_z = 0.65895$) and tangential velocity retention ratio ($k_x = 0.8520$).
   - Predicts arrival time $t^*$ at the nominal interception plane ($x = -1.47158\text{ m}$) in $< 5\ \mu\text{s}$ ($+0.11\text{ ms}$ error vs. 4000 Hz physics ground truth).
2. **Descending Phase Planner (`interception/planner.py`):**
   - Adjusts the stroke phase clock $\tau = \max(0, t - (t^* - 0.4890))$.
   - Latches the active plan prior to downswing ($t \le 0.24\text{ s}$, $x > -1.25\text{ m}$, $v_x < -0.1\text{ m/s}$), ensuring monotonic phase progression.
3. **Controller Interface & CLI (`interception/controller.py`, `continuous_feeds.py`):**
   - Adds an optional `--controller predictive-timing` CLI flag.
   - Preserves `--controller scripted` as the default.
   - Adds an optional `--velocity vx vy vz` argument for perturbation evaluation.

---

## 2. Protected Assets & Compatibility Verification

| Component | Verification Method | Status |
| :--- | :--- | :---: |
| `assets/wbc/policy.onnx` | SHA256 (`63a444122d85d286...`) | **Bit-exact match** |
| `assets/wbc/config.yaml` | SHA256 (`a646687843846d41...`) | **Bit-exact match** |
| `assets/scene.xml` | Git diff vs. `origin/main` | **0 lines modified** |
| Native Actuator Limits & Gains | `scripts/verify_assets.py` (40 portable assets) | **100% verified** |
| Baseline CLI Behavior | `python -m g1_ping_pong_lab.continuous_feeds` | **Bit-for-bit identical trajectory** |

*Baseline Bit-Identity Test:*
Executing `python -m g1_ping_pong_lab.continuous_feeds --feeds 4 --interval 3 --tail 12` with the modified code generates a `session.npz` with SHA256 hash `9ea754a03165765ffdc5f4ea95541d4eeb5ea22dd7f44f67368dafdf4a366a6f`, matching the run from the base commit `95dd0bd` bit-for-bit.

---

## 3. Verified Benchmark Results

All metrics are directly reproduced from `outputs/predictive_timing/benchmark-results.csv` and `outputs/predictive_timing/benchmark-summary.json`.

### 3.1 Velocity Generalization (Tested Speeds)

| Feed Velocity $v_x$ | Delta | Scripted Baseline | Predictive Timing | Observed Outcome (Predictive Timing) |
| :---: | :---: | :--- | :--- | :--- |
| **$-1.800\text{ m/s}$** | **$0.0\%$ (Nominal)** | **`valid_return`** | **`valid_return`** | Impact at $t=0.489\text{ s}$, clean table return |
| **$-1.854\text{ m/s}$** | **$+3.0\%$** | `net_contact` | **`valid_return`** | Impact advanced to $t=0.474\text{ s}$, returns over net |
| **$-1.890\text{ m/s}$** | **$+5.0\%$** | `body_contact` | **`valid_return`** | Impact advanced to $t=0.464\text{ s}$, clears pelvis |
| **$-1.926\text{ m/s}$** | **$+7.0\%$** | `wrong_first_landing` | **`valid_return`** | Impact advanced to $t=0.453\text{ s}$, lands on table |
| **$-2.016\text{ m/s}$** | **$+12.0\%$** | `floor_contact` | **`valid_return`** | Impact advanced to $t=0.430\text{ s}$, lands on table |
| **$-2.070\text{ m/s}$** | **$+15.0\%$** | `floor_contact` | `body_contact` (Fail) | Ball reaches pelvis before downswing clears |

### 3.2 Continuous Multi-Feed Endurance

| Metric | `scripted` Baseline | `predictive-timing` |
| :--- | :---: | :---: |
| **4-Feed Valid Returns** | 2 / 4 ($50.0\%$) | 2 / 4 ($50.0\%$) |
| **4-Feed Contacts** | 3 / 4 ($75.0\%$) | 3 / 4 ($75.0\%$) |
| **8-Feed Valid Returns** | 2 / 8 ($25.0\%$) | 2 / 8 ($25.0\%$) |
| **8-Feed Contacts** | 3 / 8 ($37.5\%$) | 3 / 8 ($37.5\%$) |
| **Stance Balance Accepted** | **True** | **True** |
| **Max Force Cap Ratio** | $\le 1.0000$ | $\le 1.0000$ |

---

## 4. Clear Separation of Capabilities and Limitations

### What the Predictor Demonstrably Improves:
- In the tested faster longitudinal speed conditions (+3%, +5%, +7%, +12%), it adjusts stroke phase to intercept the ball at the nominal contact plane, yielding valid returns where the baseline failed.

### What Remains Unchanged:
- **Multi-feed endurance:** Stance drift from floating-base balance reactions causes the robot to translate rearward by $> 3.5\text{ cm}$ over repeated feeds. Because the ball is delivered to fixed world coordinates, feeds on later cycles miss the robot's physical reach. Predictive timing alone does not resolve this spatial drift.
- **Slower feeds ($-5\%$ to $-15\%$):** Balls served slower bounce short and drop below the static arm reach.

### What Has Not Been Tested:
- Topspin or backspin Magnus aerodynamics (ball flight is modeled purely ballistically).
- High lateral angles ($|v_y| > 0.05\text{ m/s}$), which require racket yaw reorientation rather than longitudinal timing.
- Continuous rally with bidirectional ball returns.

### Why Spatial / Torso Compensation Remains Separate:
We conducted separate exploratory research using 6D orientation-constrained differential IK on branch `feat/predictive-torso-interception`. While 6D IK achieved 100% physical ball contact (4/4) across 4 feeds, valid returns remained 1/4 because arm extension shifts whole-body center-of-mass balance, causing returns to sail long of the table. True rally endurance will require active base stepping rather than arm reaching alone. Keeping the spatial compensation experimental ensures that this PR remains narrowly scoped, minimal, and verifiable.

---

## 5. Reproduction Instructions

```bash
# 1. Run complete test suite (37 passed, 2 skipped)
python -m pytest -rs

# 2. Run the automated timing benchmark suite
python tools/benchmark_predictive_timing.py

# 3. Run individual sessions:
# Baseline scripted mode (default):
python -m g1_ping_pong_lab.continuous_feeds --feeds 4 --output outputs/eval-scripted

# Predictive timing mode:
python -m g1_ping_pong_lab.continuous_feeds --feeds 4 --controller predictive-timing --output outputs/eval-timing

# Faster feed evaluation (+5% speed):
python -m g1_ping_pong_lab.continuous_feeds --controller predictive-timing --velocity -1.89 0.0 0.5 --output outputs/eval-fast
```

---

## 6. Draft Message to Maintainer (Alpamys)

*Note: Do not send until the branch is pushed to your personal fork.*

```markdown
Hi Alpamys,

Following our earlier conversation, I have prepared a scoped, self-contained contribution on branch `feat/predictive-impact-timing` that adds an analytical ball-flight and impact-timing predictor to the arm trajectory pipeline.

### Summary of Changes:
1. **Piecewise Analytical Predictor (`interception/predictor.py`):** Evaluates ballistic flight and table bounce using calibrated restitution parameters ($e_z = 0.65895$, tangential velocity retention ratio $k_x = 0.8520$) to predict arrival time at the interception plane in $< 5\ \mu\text{s}$ ($+0.11\text{ ms}$ error vs. 4000 Hz simulation ground truth).
2. **Descending Phase Planner (`interception/planner.py`):** Dynamically adjusts the stroke phase clock $\tau = \max(0, t - (t^* - 0.4890))$ with plan latching prior to downswing ($t \le 0.24\text{ s}$).
3. **CLI Integration (`continuous_feeds.py`):** Adds an optional `--controller predictive-timing` flag and `--velocity vx vy vz` argument. The original scripted controller remains the default.
4. **Benchmark Suite (`tools/benchmark_predictive_timing.py`):** Compares both controllers across speed sweeps, multi-feed endurance, and variation trials.

### Empirical Results:
- **Faster Feeds (+3% to +12%):** In our evaluations at $+3\%$, $+5\%$, $+7\%$, and $+12\%$ speed, predictive timing achieved a valid return on each tested condition, whereas the scripted baseline returned 0/4. At $+15\%$, both controllers failed (the ball clipped the pelvis).
- **Endurance Invariance:** Multi-feed endurance on fixed feeds remains 2/4 and 2/8 for both controllers. This confirmed that endurance failure is driven by floating-base stance drift ($> 3.5\text{ cm}$ rearward), which requires active base stepping rather than arrival timing.
- **Exploratory Stance Work:** We also tested 6D differential IK for stance-drift compensation on a separate branch (`feat/predictive-torso-interception`). While it achieved 100% contact coverage (4/4), returns past Feed 1 missed the table due to altered COM momentum balance. We kept that work separate to ensure this PR remains tightly scoped and robust.

### Compatibility:
- Pretrained WBC policy (`policy.onnx`), `config.yaml`, and `scene.xml` are completely untouched.
- Default execution generates trajectory data (`session.npz`) matching the main branch commit `95dd0bd` bit-for-bit.
- All 37 non-render tests pass (`pytest -rs`), and all 40 portable assets verify clean.

The branch is published on my fork for your review whenever convenient:
https://github.com/<your-username>/G1-Ping-Pong-Lab/tree/feat/predictive-impact-timing

Best regards,  
Pernav
```
