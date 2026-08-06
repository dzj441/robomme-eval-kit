# RoboMME memory interventions — running notes

Same checkpoint throughout: `perceptual-framesamp-modul`, ckpt 79999, seed 7,
16 tasks x 50 episodes. Baseline for comparison is `eval_out/ultra3_seed7`
(AVG 46.5 on this seed; 45.5 +- 0.9 over three seeds, vs the paper's 44.51).

## Offline analyses (no GPU, complete)

### 1. Uniform sampling does not drop the evidence

`even_sampling_indices` is `np.linspace(0, step_idx, 32)` — content-agnostic and
fixed-size. The obvious suspicion is that it skips the decisive frame. Three
measurements say it does not.

**Subgoal-segment coverage** (`sampling_coverage.py`, 20 episodes/task from the
h5 release, scored at the last step where the history is longest):

    missed segments: 0.0% on all 16 tasks

Every behavioural segment gets at least one sampled frame, including the longest
tasks. Segments run 32-96 frames against a sampling interval of 6-39.

**Visual-state coverage** (`visual_event_coverage.py`, label-free: spikes in the
frame-to-frame distance of the cached SigLIP embeddings, sigma=2). Stratified by
how long a state persists, because a one-frame blip between two detector spikes
carries no fact and missing it means nothing:

    states lasting >=10 frames:   0.0-12.7% missed  (11 of 16 tasks at 0.0%)
    states lasting  <10 frames:  60-96%    missed

The two worst >=10f rates are the two longest tasks — VideoPlaceOrder 11.2%
(interval 39.1) and BinFill 12.7% (interval 20.7).

**Linear decodability** (`probe_decodability.py`, PickHighlight, 100 episodes,
4 decision points each spread across the execution phase, folds grouped by
episode). Predict the highlighted colour from the cached features:

    majority baseline                        36.0%
    cur1    current frame only               46.2% +- 3.8
    samp32  exactly what FrameSamp keeps     60.8% +- 8.3
    full    every 2nd frame of the history   62.3% +- 7.2
    past32  32 frames before the decision    66.0% +- 5.8

`cur1` well below `samp32` rules out the probe simply reading the current
observation — the history contributes ~15pp. `samp32` matching `full` is the
result that matters: the sampler costs nothing a linear readout can detect.

The policy itself scores **21.3%** on PickHighlight, below the 36% you would get
by always guessing the most common colour.

Caveats worth keeping: the probe pools mean+max over frames, so it is a lower
bound on what is extractable; 60% is not high in absolute terms; task success
needs more than naming the colour; and only two tasks have a label recoverable
from the subgoal text.

### 2. The same probe on MoveCube, and what the pair implies

MoveCube states its answer in the subgoal too — the manner is one of `place`,
`push` or `hook` — and it is the best-performing memory-conditioned task at
83.3%. Anchoring the decision at the first step that *names* the manner turned
out to be wrong: by then the arm is already holding the peg, and `cur1` alone
scored 93.8%. Re-anchored at `exec_start_idx`, the instant execution begins and
before any manner is expressed:

    majority baseline                        35.0%
    cur1    current frame only               39.0% +- 9.2
    samp32  exactly what FrameSamp keeps     98.0% +- 2.4
    full    every 2nd frame of the history   99.0% +- 2.0
    past32  32 frames before the decision    99.0% +- 2.0

`cur1` at chance confirms the answer is genuinely absent from the present, and
`samp32` at 98% that the memory carries it near-perfectly.

Putting the two tasks side by side:

| task | cur1 | samp32 | policy | shortfall |
|---|---|---|---|---|
| MoveCube | 39.0 | 98.0 | 83.3 | -15 |
| PickHighlight | 46.2 | 60.8 | 21.3 | -40 |

A worry about the pooling: `frame_feats` averaged the 4x4 patch grid away, while
the model receives the patches intact, so the probe was strictly weaker than the
model's own input — and "which cube was highlighted" is partly a question about
*where*. Re-running PickHighlight with the grid kept (65536-d, PCA'd inside each
fold) changes nothing:

    cur1 45.0   samp32 61.5   full 61.3   past32 66.0

So the ~61% is not an artifact of pooling. That makes the two tasks differ in a
second way, and the honest reading is that binding tasks carry *two* stacked
deficits rather than one: the memory encodes the binding only partially (61% vs
MoveCube's 98%), and the policy converts even that partial signal badly (21%).

The cached features also carry 2x2 and 8x8 grids, so the obvious follow-up is
whether the deployed 4x4 is simply too coarse to localise a small cube. It is
not — PickHighlight's decodability is flat across a 64-fold change in tokens per
frame:

    grid    tokens/frame   cur1   samp32   full   past32
    pooled       1         46.2    60.8    62.3    66.0
    2x2          4         43.3    62.3    62.3    67.0
    4x4         16  <-     45.0    61.5    61.3    66.0
    8x8         64         46.0    63.0    63.3    68.0

MoveCube is at ceiling either way (98.0 at 4x4, 99.0 at 8x8). So the "coarse
spatial pooling loses the binding" hypothesis is wrong and withdrawn.

That leaves three candidate causes ruled out for the binding deficit — temporal
sampling, spatial resolution, and the current-frame confound — and points at the
frozen per-frame SigLIP2 embedding itself: it linearly separates a scene-level
gist ("which of three manners") far better than an object-level identity ("which
cube was highlighted, when"). The implied direction is object- or region-level
memory tokens, not more frames and not more patches.

It also inverts one knob: since 4 tokens per frame decode as well as 16,
`token_per_image` could be cut fourfold and spent on 4x the frames instead —
which is exactly what `budget128` tests from the other side.

Stated precisely, this is about *linear* availability. A single attention read-out
is approximately a linear query, which is why that is the relevant notion here,
but a nonlinear probe could well do better.

A first attempt to settle that was inconclusive: an MLP (256,64) head scored 53.0
on PickHighlight against the linear 61.5, but at 300 samples that is more likely
overfitting than absence of signal.

Re-run at 20 decision points per episode — 2000 samples, still grouped by
episode, `cur1`/`samp32` only to keep the loading affordable:

    linear   cur1 48.0 +- 3.0    samp32 61.5 +- 6.8
    mlp      cur1 46.4 +- 4.4    samp32 54.3 +- 5.8

The linear estimate is unchanged from the 400-sample run (61.5 both times), and
the MLP still does not beat it with five times the data. So the ~61% reads as a
ceiling of this feature space rather than of the decoder's linearity.

That is corroborated by the paper's own ablation from the other direction: its
three integration mechanisms — context, modulator, expert — are all *readout*
variants over the same FrameSamp representation, and they span only 34.5-44.5,
none of them close to the 84.1 an oracle memory reaches. Changing how the memory
is read does not move the ceiling because the ceiling is in what is stored.

(One nonlinear family is not every nonlinear family, so this is evidence rather
than proof.)

Both have the answer in memory to some degree and neither loses it to sampling.
What separates them is the *kind* of answer. MoveCube's is a global property of the whole demo —
which of three manners was used. PickHighlight's is an object-level binding —
which cube was highlighted, at which moment. The policy converts the first into
task success and largely fails to convert the second, scoring below the 36% it
would get by always naming the most common colour.

Hypothesis this suggests: the modulation channel carries global scene properties
but not object bindings. That is testable — it predicts `off` should cost little
on tasks whose answer is a binding and a lot on tasks whose answer is global.

Committing to the split before the full `off` numbers land, so the prediction can
actually fail:

    global  (a property of the history as a whole, or a count of one's own acts)
            BinFill, StopCube, PickXtimes, SwingXtimes, MoveCube,
            PatternLock, RouteStick
    binding (which object, which container, which target, which instance)
            VideoUnmask, ButtonUnmask, VideoUnmaskSwap, ButtonUnmaskSwap,
            PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder
    neither InsertPeg — manipulation-bound (Oracle reaches only 15.6), and a
            useful control: `off` should barely move it either way

Partial `off` numbers at the time of writing already run: PickXtimes -66,
SwingXtimes -67, StopCube -36, BinFill -22 (all global, all large) against
VideoUnmask -3 (binding, small) — but ButtonUnmask was at -21 on 19 of 50
episodes, which would be a counterexample if it holds up.

### 3. The same split separates MemER from FrameSamp in the paper's own numbers

MemER is the strongest non-oracle competitor in Table 3 and the one method that
beats FrameSamp+Modul on the tasks our failure analysis flagged. Scoring its
published per-task numbers against ours under the split above:

    global tasks (n=7):  MemER - ours = -25.2 mean, MemER wins 1/7
    binding tasks (n=8): MemER - ours = +15.8 mean, MemER wins 5/8

The three largest positive gaps are all binding tasks (ButtonUnmask +52.7,
VideoUnmask +51.3, PickHighlight +49.3); the three largest negative gaps are all
global (RouteStick -56.0, StopCube -48.0, PatternLock -42.0). Two methods with
opposite strengths, separated along the axis the probe suggested.

Two exceptions are informative rather than damaging. BinFill is nominally a count
but "which cubes have I already binned" is a binding, and MemER wins it. The
VideoPlace pair are bindings but strongly temporal — "the target right before the
button", "the first target" — and dense uniform sampling preserves order in a way
sparse keyframes do not, which is where our variant keeps its advantage.

This is corroboration, not proof: MemER is a different architecture (VLM keyframe
selection driving discrete keyframe actions), so more differs between the columns
than the kind of answer, and its numbers come from the paper's 9 runs against our
3 seeds.

(PickHighlight's success criterion is a conjunction of several picks while the
probe is a single 3-way choice, so the two shortfalls are not strictly
comparable in magnitude.)

## Test-time interventions (GPU queue)

Engaged via `ROBOMME_MEM_MODE` through `serve_policy_mem.py`, which patches
`MemoryBuffer` before the stock server starts. Verified per mode: budget 512->2048
gives 128 frames, 512->128 gives 8, primacy front-loads 12 of 32 into the first
eighth. Every run checks that all 16 servers logged the patch.

| mode | change | question |
|---|---|---|
| `off` | memory zeroed and masked | what is memory worth under these weights? |
<!-- `off` audited numerically against the real modules (CPU, freshly initialised
     params for the attention, trained params for the norm):
       real memory, mask=True    max|out| 0.343
       real memory, mask=False   max|out| 0.144   <- masking alone leaves signal
       zero memory, mask=False   max|out| 0       <- what `off` does
       zero memory, mask=True    max|out| 0
     So masking alone is genuinely insufficient (uniform softmax over real v),
     but zeroing alone is sufficient -- the mask is redundant, not required, and
     the earlier "must do both" was imprecise. Doing both is still correct.

     With cond=0 the modulation is Dense's bias, and in the trained checkpoint
     that bias is NOT zero: mem_rms_norm_ffn/Dense_0/bias is (18, 2048) with
     scale rms 0.075 and shift rms 0.022, i.e. a fixed affine applied at each of
     the 18 action-expert layers. `off` therefore is not "the branch removed" but
     "the branch pinned to a constant" -- and cond=0 is itself outside the Dense's
     training distribution, so that constant is an extrapolation rather than a
     neutral default. Two OOD effects, on the memory tensor and on the Dense
     input, and the 31.5-point drop cannot be apportioned between them. -->

<!-- On the spatial-resolution sweep and RoPE: the probe never runs the model, so
     its numbers carry no RoPE dependence. But acting on them does. Changing
     token_per_image changes mem_len = frames x tokens, and the action expert's
     query RoPE starts at mem_len, so deploying 8x8 (32 x 64 = 2048) would walk
     straight into the budget128 failure. Hence frames128 = 128 x 4 = 512. -->

| `timeshuf` | frame order and pos_emb independently permuted | is the ordering used at all? |
| `budget128` | 32 -> 128 frames | is 32 too few? |
| `budget8` | 32 -> 8 frames | is 32 more than enough? |
| `primacy` | same 32, front-loaded | does reallocating help? |

`off`/`budget8`/`budget128` together give a 0/8/32/128 dose-response curve.

`primacy` was demoted to last after the coverage results came in: with nothing
being missed at segment granularity, reallocating the budget is the weaker
hypothesis.

### Results

    task              suite   base      off   timeshuf  budget128  budget8
    AVG                       46.5     15.0     17.0      20.0      17.9
      Counting                69.5     22.0     19.0      25.0      17.0
      Permanence              25.5     11.5     16.0      22.0      17.5
      Reference               37.5     14.0     18.5      18.5      18.5
      Imitation               53.5     12.5     14.5      14.5      18.5
    timeout share              6.3%    32.4%    24.5%     10.8%     18.5%

Every intervention lands between 15 and 20 against a baseline of 46.5, and the
paper's memory-free pi0.5 scores 17.93. Including `budget128`, which is given
*four times more* of exactly the evidence the baseline gets and does no better
than having none at all. Timeouts rise two- to fivefold, so the perturbed policy
is getting stuck rather than deciding wrongly.

That pattern rules out the reading the batch was designed to produce. `off`
cannot be quoted as "memory is worth 31.5 points" when `timeshuf` — every frame
present, every pixel intact, only the order destroyed — costs nearly the same.

**The dose-response framing is withdrawn entirely**, and the reason is in the
attention module rather than in the data:

    q_positions = jnp.arange(mem_len, x_len + mem_len)
    k_positions = jnp.arange(mem_len)

The action expert's *query* positions start at the memory length. At the trained
setting mem_len is 32 frames x 16 tokens = 512, so queries sit at 512..512+x_len;
under `budget128` they jump to 2048, a RoPE regime the model never saw. So
`budget8` and `budget128` do not measure whether 32 frames is the right number.
They measure what happens when the queries' positional encoding is moved, and
the answer is that it breaks.

That is worth stating as a finding in its own right: this design ties the action
expert's positional encoding to the memory length, which makes the memory size a
baked-in architectural constant rather than a hyperparameter. A memory that
cannot be given more frames without breaking is not one that can be scaled.

It also dictates the experiment that *would* answer the question. Dropping
`token_per_image` from 16 to 4 gives 128 frames x 4 tokens = 512 memory tokens —
the same mem_len, the same RoPE, four times the temporal resolution. The probe
above already shows 2x2 features decode as well as 4x4 on both labelled tasks, so
the spatial cost is close to nothing. That run is the clear next step.

### The control this batch was missing — now run, and clean

All four interventions went through `serve_policy_mem.py` while the baseline used
the stock server, and all four landed at 15-20. A side effect of the wrapper
would produce exactly that pattern, so `ROBOMME_MEM_MODE=baseline` was run
through the same wrapper on the four Counting tasks:

    task           stock base  via wrapper     off
    BinFill              46.0         42.0    24.0
    StopCube             44.0         44.0     8.0
    PickXtimes           94.0         90.0    28.0
    SwingXtimes          94.0         96.0    28.0
    mean                 69.5         68.0    22.0

68.0 against 69.5, inside the +-7pp per-task standard error, and nowhere near
`off`'s 22.0. The wrapper is not implicated; the drops are real.

### `repeatcur`: the in-distribution floor, and what it settles

`off` was confounded twice over -- zeros in the memory tensor, and zeros into a
Dense that never saw them -- so its 31.5-point drop could not be apportioned
between "history removed" and "module pushed off-distribution". `repeatcur` fills
every slot with the newest sampled frame, which is the current observation:
every entry is a real embedding of the kind training used, `mem_mod_vec` lands
in a plausible region, and pos_emb keeps its true timestamps. What is removed is
the history, since the current frame already reaches the VLM anyway.

    AVG        base 46.5    off 15.0    repeatcur 14.6
    timeout %       6.3          32.4             35.1

14.6 against 15.0, well inside the ~1.8pp standard error on AVG. **The OOD worry
was legitimate as method and empirically worth almost nothing.** Two perturbations
as different in their off-distribution character as all-zeros and
all-real-current-frame land on the same number; what they share is the absence of
history, so that is what the drop measures.

So the conclusion upgrades from "the memory branch is not vestigial" to **history
is worth about 32 points, measured with a control that stays in distribution
throughout**. For calibration our no-history floor of 14.6 sits *below* the
paper's separately trained memory-free pi0.5 at 17.93 -- a model trained without
memory learns to cope, ours learned to depend on it and was then deprived.

Normalising every intervention against that floor:

    floor (repeatcur) 14.6, ceiling (baseline) 46.5, span 31.9

    off        15.0     1.3% of the memory's value retained
    timeshuf   17.0     7.5%
    budget8    17.2     8.2%
    budget128  20.0    16.9%
    frames128  30.8    50.8%   <- the only one that keeps half

Everything but `frames128` sits within 2.5 points of the floor, which also settles
`timeshuf`: destroying the temporal order costs nearly as much as having no
history at all.

One caveat against over-reading the agreement. Per task the two diverge, in both
directions and by more than the ~7pp per-task standard error:

    MoveCube     off 42.0   repeatcur 18.0   -24
    VideoUnmask  off 30.0   repeatcur 18.0   -12
    BinFill      off 24.0   repeatcur 32.0    +8
    SwingXtimes  off 28.0   repeatcur 34.0    +6

The matching averages are partly offsetting differences. A reading worth testing
rather than asserting: `repeatcur` claims "every past moment looked like now",
which on MoveCube is confidently false -- the current frame shows neither peg nor
manner -- while `off` merely supplies a neutral constant. If that holds, a wrong
memory really can be worse than no memory, which is the quantity a cross-episode
swap was meant to measure.

### What the batch does and does not establish

Establishes:

- The memory is load-bearing. Removing it costs 31.5 points of AVG, and on
  PickHighlight, VideoRepick, ButtonUnmaskSwap and RouteStick it takes the policy
  to exactly zero.
- The mechanism cannot be resized. `budget128` supplies four times the evidence
  and scores 20.0 against the baseline's 46.5, for the RoPE reason above. Memory
  size is an architectural constant here, not a hyperparameter.
- Counting rides on the slot index specifically. Under `budget128` the Counting
  suite falls 44.5 while Permanence falls 3.5 — changing the number of slots
  destroys counting and leaves "what was under there" almost intact, which is
  what a slot-index RoPE predicts.

Does not establish:

- Whether the *temporal ordering* specifically carries the value. `timeshuf` lands
  at 17.0, i.e. on the no-history floor, but it is still an off-distribution
  manipulation, and `repeatcur` does not rule that out for it the way it does for
  `off` -- a permuted memory is not something training ever produced. Order might
  carry nearly everything, or shuffling might simply break the module.
- Any dose-response in the number of frames, for the reason given above.

Superseded by `repeatcur`: the earlier reading that `off`'s drop "cannot be
apportioned between lost history and off-distribution input" no longer holds. It
can, and the off-distribution share is approximately zero.

The prediction registered earlier — that `off` would cost little on binding tasks
— is **falsified**. PickHighlight goes 20.0 -> 0.0 and VideoRepick 28.0 -> 0.0.
Binding tasks depend on the memory as completely as any other; they just convert
less of it into success.
