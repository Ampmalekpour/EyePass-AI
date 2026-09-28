"""
plate_tracker.py  -  ByteTrack for the plate system  ("plates are not forgotten" edition, v2)
====================================================================================

WHY TRACKS WERE BEING LOST  (read this before touching the knobs)
------------------------------------------------------------------------------------
Symptom: a car is detected with good confidence (s0.86), the overlay says
`dets 1 | tracks 0(+1 ghost)`, the track is `age 2f`, and the pipeline then
logs `DROPPED: seen 7<8`. The plate is forgotten even though YOLO never
stopped seeing the car.

Stock ByteTrack has a two-tier lifecycle: a new track is "unconfirmed" and
must match on the very next step or it is deleted. At ~9 engine fps on a
25-30 fps stream every tracker step is ~3 camera frames of motion, and stock
ByteTrack (a) never predicts unconfirmed tracks, (b) gives new tracks zero
velocity and (c) kills them on the first miss - so fast cars cycle through
fresh ids and never collect 8 seen frames.

A speed bump is the worst case of the same thing. The car brakes before it
(constant-velocity prediction overshoots), the box jolts up/down and changes
shape as the car pitches (IoU collapses), and motion blur drops the detector
confidence below track_thresh or loses the car for a step.

v1 FIXES  (kept)
------------------------------------------------------------------------------------
  FIX 1  unconfirmed tracks are Kalman-predicted          -> predict_unconfirmed
  FIX 2  new tracks get a velocity from their first displacement
                                  -> seed_velocity_on_first_update / new_track_vel_std_scale
  FIX 3  unconfirmed tracks get a grace period             -> unconfirmed_max_miss
  FIX 4  real elapsed time per step                        -> update(..., dt=N)
  FIX 5  overlap-free recovery association                 -> recovery_*
  FIX 6  camera-jolt compensation from box evidence        -> gmc_*
  FIX 7  bounded removed_stracks                           -> max_removed_history
  FIX 8  tagged logging + get_stats()

v2  BUG FIXES
------------------------------------------------------------------------------------
  B1  Velocity seeding counted the motion twice. The displacement was ADDED to
      the velocity and then the Kalman update added its own correction for the
      same innovation, so a new track left with ~1.7x its real speed and
      overshot on exactly the step that decides whether it survives.
      Now the filter updates first and the seed OVERWRITES the velocity.
  B2  Lost tracks were only ever compared at their *predicted* box, which keeps
      flying forward at the old speed. Cars slow down for bumps, so the
      prediction ran away from the car. Recovery now scores both the predicted
      box and the last box the detector actually saw, and keeps the better.
  B3  The camera-shake vote counted (track, detection) PAIRS. Two fragments of
      the SAME car (an old lost track + a new one) produced two agreeing votes
      and a single moving car was mistaken for a camera jolt. Support is now
      counted in distinct tracks AND distinct detections. The matched-track
      estimate also fired on almost every frame (any shift > 1 px, fed by
      brand-new tracks whose "residual" is just their own motion); it now needs
      established tracks, agreeing residuals and a shift of >= 8% of a box
      height.                                            -> gmc_min_shift_ratio
  B4  Unit mix-up: with dt given in camera frames, the recovery gate grew per
      camera frame instead of per step (~3x too fast at 9 fps) and
      reseed_after_gap fired after a single miss. Gates now use tracker steps;
      only the motion model uses dt.
  B5  A detection with score exactly == track_thresh was silently discarded
      (neither high nor low tier).
  B6  Wide-gate recovery for lost tracks ran BEFORE young tracks got their
      plain-IoU match, so a lost track could steal a young track's detection.
      Young tracks now get their IoU match first, and recovery runs once,
      jointly, for everything that is left.

v2  BUMP ROBUSTNESS
------------------------------------------------------------------------------------
  R1  Low-score detections (motion blur on the bump) can now keep young tracks
      alive, and a strict low-score recovery pass rescues tracked cars whose
      blurred box also jumped.                              -> low_recovery_*
  R2  Observation-centric re-update (from OC-SORT). When a track comes back
      after missing steps its coasted velocity is garbage; the gap is replayed
      with interpolated observations so the velocity is right immediately.
                                                                -> oru_enabled
  R3  Tracks that left the frame are removed instead of lingering for the full
      buffer with a wide gate, where they could grab the next car arriving at
      the same edge (this was the main source of ID switches).
                                                              -> remove_exited
  R4  Recovery refuses pairs whose size differs by more than 2x.
                                                     -> recovery_min_size_ratio

LOG TAGS
------------------------------------------------------------------------------------
  [TRK-NEW] [TRK-CONFIRM] [TRK-LOST] [TRK-REFIND via=iou|low|recovery|
  recovery-low|recovery-young] [TRK-COAST] [TRK-REMOVE] [GMC]
  `[TRK-REMOVE] reason=unconfirmed_expired` is literally a forgotten plate.

TO GO BACK TO STOCK BYTETRACK BEHAVIOUR (for an A/B comparison) set:
    predict_unconfirmed=False, seed_velocity_on_first_update=False,
    new_track_vel_std_scale=1.0, unconfirmed_thresh=0.7, unconfirmed_max_miss=0,
    recovery_enabled=False, low_recovery_enabled=False, gmc_enabled=False,
    oru_enabled=False, remove_exited=False
...and call update() without dt.

INTERFACE  (unchanged - drop-in for yolox.tracker.byte_tracker)
------------------------------------------------------------------------------------
    BYTETracker(args, frame_rate=30, name=None)
        args needs .track_thresh .match_thresh .track_buffer .mot20
        every other knob is read with getattr() and has a default
    tracker.update(output_results, img_info, img_size, dt=None) -> list[STrack]
        output_results: Nx6  [x1, y1, x2, y2, score, class_id]
        dt: optional, how many camera frames elapsed since the last update
    each returned STrack exposes .track_id .score .flag_fdf .detbb .tlwh .tlbr
    `lap` is used when installed; otherwise scipy does the assignment.
"""

import logging
from collections import OrderedDict, deque

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment as _scipy_lsa

try:
    import lap as _lap
except Exception:  # pragma: no cover
    _lap = None

try:
    from cython_bbox import bbox_overlaps as _bbox_ious_cython
except Exception:  # pragma: no cover - fallback keeps the file usable anywhere
    _bbox_ious_cython = None


logger = logging.getLogger("plate_tracker")

# Cost value used for pairs that a hard gate rejected. Must stay far above any
# association threshold so the assignment can never pick it.
_REJECT = 1e5


# ====================================================================
# Track state
# ====================================================================
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


# ====================================================================
# Kalman filter  (8-state constant velocity: x, y, a, h + velocities)
# predict()/multi_predict() take a real dt; initiate() can widen the
# initial velocity uncertainty.
# ====================================================================
class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.
        self._ndim = ndim

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # motion matrices for non-unit dt, built on demand and cached
        self._motion_mat_cache = {1.0: self._motion_mat}

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def motion_mat(self, dt=1.0):
        """Constant-velocity transition matrix for an arbitrary step size."""
        dt = float(dt)
        mat = self._motion_mat_cache.get(dt)
        if mat is None:
            if len(self._motion_mat_cache) > 256:   # dt from timestamps can be anything
                self._motion_mat_cache = {1.0: self._motion_mat}
            ndim = self._ndim
            mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                mat[i, ndim + i] = dt
            self._motion_mat_cache[dt] = mat
        return mat

    def initiate(self, measurement, vel_std_scale=1.0):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        vs = float(vel_std_scale)
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3] * vs,
            10 * self._std_weight_velocity * measurement[3] * vs,
            1e-5,
            10 * self._std_weight_velocity * measurement[3] * vs]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, dt=1.0):
        dt = float(dt)
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * dt

        mm = self.motion_mat(dt)
        mean = np.dot(mean, mm.T)
        covariance = np.linalg.multi_dot((mm, covariance, mm.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance, dt=1.0):
        dt = float(dt)
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T * dt

        motion_cov = np.zeros((len(mean), 8, 8))
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr

        mm = self.motion_mat(dt)
        mean = np.dot(mean, mm.T)
        left = np.dot(mm, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, mm.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('invalid distance metric')


# ====================================================================
# Assignment / cost helpers
# ====================================================================
def linear_assignment(cost_matrix, thresh):
    """
    Min-cost matching that leaves any pair costing more than `thresh`
    unmatched (same semantics as lap.lapjv(extend_cost=True, cost_limit=thresh)).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        cost_matrix = cost_matrix.reshape(0, 0)
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(n, dtype=int),
                np.arange(m, dtype=int))

    if _lap is not None:
        _, x, y = _lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    else:
        # augmented square problem: every row/col may go to a dummy at thresh/2
        half = float(thresh) / 2.0
        C = np.zeros((n + m, n + m), dtype=float)
        C[:n, :m] = np.minimum(cost_matrix, float(thresh) + 1.0)
        C[:n, m:] = half
        C[n:, :m] = half
        r, c = _scipy_lsa(C)
        x = -np.ones(n, dtype=int)
        y = -np.ones(m, dtype=int)
        for i, j in zip(r, c):
            if i < n and j < m and cost_matrix[i, j] <= thresh:
                x[i] = j
                y[j] = i

    matches = np.asarray([[i, int(j)] for i, j in enumerate(x) if j >= 0],
                         dtype=int).reshape(-1, 2)
    unmatched_a = np.where(np.asarray(x) < 0)[0]
    unmatched_b = np.where(np.asarray(y) < 0)[0]
    return matches, unmatched_a, unmatched_b


def fuse_score(cost_matrix, detections):
    """Fold detection confidence into the IoU cost (stock ByteTrack)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim


def _ious_numpy(atlbrs, btlbrs):
    a = np.asarray(atlbrs, dtype=float).reshape(-1, 4)
    b = np.asarray(btlbrs, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def ious(atlbrs, btlbrs):
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if out.size == 0:
        return out
    if _bbox_ious_cython is not None:
        return _bbox_ious_cython(
            np.ascontiguousarray(atlbrs, dtype=float),
            np.ascontiguousarray(btlbrs, dtype=float))
    return _ious_numpy(atlbrs, btlbrs)


def iou_distance(atracks, btracks):
    """Cost = 1 - IoU. Accepts either STrack lists or raw Nx4 tlbr arrays."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    return 1 - ious(atlbrs, btlbrs)


def iou_distance_boxes(a_boxes, b_boxes):
    """Cost = 1 - IoU between two Nx4 / Mx4 tlbr arrays (no STrack needed)."""
    a = np.asarray(a_boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(b_boxes, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    return 1.0 - ious(a, b)


def fuse_score_array(cost_matrix, det_scores):
    """fuse_score() variant that takes a plain score array."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    s = np.asarray(det_scores, dtype=float).reshape(1, -1)
    return 1 - (iou_sim * s)


def expand_boxes(boxes, ratio):
    """Grow each box by `ratio` of its own size (half on each side)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if ratio <= 0 or len(b) == 0:
        return b.copy()
    out = b.copy()
    w = (b[:, 2] - b[:, 0]) * ratio * 0.5
    h = (b[:, 3] - b[:, 1]) * ratio * 0.5
    out[:, 0] -= w
    out[:, 2] += w
    out[:, 1] -= h
    out[:, 3] += h
    return out


def recovery_distance(track_boxes,
                      det_boxes,
                      frames_missing,
                      track_classes=None,
                      det_classes=None,
                      expansion=0.5,
                      base_radius=1.5,
                      radius_growth=0.4,
                      max_radius=4.0,
                      shape_weight=0.3,
                      class_penalty=0.15,
                      min_size_ratio=0.5):
    """
    Overlap-free association cost in [0, 1], or _REJECT for gated-out pairs.

      proximity : centre distance in mean box heights, divided by a gate radius
                  that grows with the number of tracker STEPS the track has
                  been missing.
      shape     : width/height similarity.
      expanded  : IoU after inflating both boxes by `expansion`; wins over the
      IoU         proximity term when the inflated boxes overlap.

    Pairs whose width or height differ by more than 1/min_size_ratio are
    rejected outright (v2, R4): a car does not double in size in a few steps.
    """
    tb = np.asarray(track_boxes, dtype=float).reshape(-1, 4)
    db = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    n, m = len(tb), len(db)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    tcx = (tb[:, 0] + tb[:, 2]) * 0.5
    tcy = (tb[:, 1] + tb[:, 3]) * 0.5
    tw = np.maximum(tb[:, 2] - tb[:, 0], 1.0)
    th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)

    dcx = (db[:, 0] + db[:, 2]) * 0.5
    dcy = (db[:, 1] + db[:, 3]) * 0.5
    dw = np.maximum(db[:, 2] - db[:, 0], 1.0)
    dh = np.maximum(db[:, 3] - db[:, 1], 1.0)

    dist = np.sqrt((tcx[:, None] - dcx[None, :]) ** 2 +
                   (tcy[:, None] - dcy[None, :]) ** 2)
    scale = np.maximum(0.5 * (th[:, None] + dh[None, :]), 1.0)
    ndist = dist / scale

    missing = np.asarray(frames_missing, dtype=float).reshape(-1, 1)
    radius = np.minimum(base_radius + radius_growth * missing, max_radius)
    radius = np.maximum(radius, 1e-6)
    prox_cost = np.clip(ndist / radius, 0.0, 1.0)

    w_sim = np.minimum(tw[:, None], dw[None, :]) / np.maximum(tw[:, None], dw[None, :])
    h_sim = np.minimum(th[:, None], dh[None, :]) / np.maximum(th[:, None], dh[None, :])
    shape_cost = 1.0 - (w_sim * h_sim)

    cost = (1.0 - shape_weight) * prox_cost + shape_weight * shape_cost

    eiou = ious(expand_boxes(tb, expansion), expand_boxes(db, expansion))
    eiou = np.asarray(eiou, dtype=float).reshape(n, m)
    cost = np.minimum(cost, 1.0 - eiou)

    if class_penalty > 0 and track_classes is not None and det_classes is not None:
        tc = np.asarray(track_classes).reshape(-1, 1)
        dc = np.asarray(det_classes).reshape(1, -1)
        cost = cost + class_penalty * (tc != dc).astype(float)

    cost = np.clip(cost, 0.0, 1.0)

    gate = (ndist > radius) & (eiou <= 0.0)
    if min_size_ratio > 0:
        gate |= (w_sim < min_size_ratio) | (h_sim < min_size_ratio)
    cost[gate] = _REJECT
    return cost


def estimate_global_shift(residuals, min_pairs=2, max_shift=None, min_shift=0.0):
    """
    GMC-lite from tracks that DID match: median of (det_centre - predicted
    centre). Only well-established tracks feed this (see BYTETracker.update),
    because a young track's residual is its own motion, not the camera's.

    v2: the residuals must AGREE (spread well below the shift itself) and the
    shift must exceed `min_shift`. Otherwise ordinary prediction lag of a few
    cars braking together was being reported as a camera jolt on most frames.
    """
    if residuals is None or len(residuals) < max(1, int(min_pairs)):
        return None
    r = np.asarray(residuals, dtype=float).reshape(-1, 2)
    dx = float(np.median(r[:, 0]))
    dy = float(np.median(r[:, 1]))
    mag = float(np.hypot(dx, dy))
    if mag < max(min_shift, 1e-6):
        return None
    if max_shift is not None and mag > max_shift:
        return None
    spread = float(np.median(np.hypot(r[:, 0] - dx, r[:, 1] - dy)))
    if spread > 0.5 * mag:
        return None
    return dx, dy


def vote_global_shift(track_boxes, det_boxes, min_support=2, max_shift=None,
                      size_tol=0.45, cluster_tol_ratio=0.6):
    """
    GMC-lite from tracks that did NOT match: every plausible (track, detection)
    pairing votes for the offset that would align it; a real camera shift moves
    every object by the same vector.

    v2 (B3): support is counted in DISTINCT tracks and DISTINCT detections.
    Before, two fragments of one car voting for the same detection counted as
    two objects, and a single driving car looked like a camera jolt.
    """
    tb = np.asarray(track_boxes, dtype=float).reshape(-1, 4)
    db = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    n, m = len(tb), len(db)
    min_support = max(2, int(min_support))
    if n < min_support or m < min_support:
        return None

    tcx = (tb[:, 0] + tb[:, 2]) * 0.5
    tcy = (tb[:, 1] + tb[:, 3]) * 0.5
    tw = np.maximum(tb[:, 2] - tb[:, 0], 1.0)
    th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
    dcx = (db[:, 0] + db[:, 2]) * 0.5
    dcy = (db[:, 1] + db[:, 3]) * 0.5
    dw = np.maximum(db[:, 2] - db[:, 0], 1.0)
    dh = np.maximum(db[:, 3] - db[:, 1], 1.0)

    w_sim = np.minimum(tw[:, None], dw[None, :]) / np.maximum(tw[:, None], dw[None, :])
    h_sim = np.minimum(th[:, None], dh[None, :]) / np.maximum(th[:, None], dh[None, :])
    ok = (w_sim >= size_tol) & (h_sim >= size_tol)
    if not ok.any():
        return None

    ti, dj = np.nonzero(ok)
    ox = dcx[dj] - tcx[ti]
    oy = dcy[dj] - tcy[ti]
    if max_shift is not None:
        keep = np.hypot(ox, oy) <= max_shift
        ox, oy, ti, dj = ox[keep], oy[keep], ti[keep], dj[keep]
    if len(ox) < min_support:
        return None
    _MAX_VOTES = 600
    if len(ox) > _MAX_VOTES:
        pick = np.linspace(0, len(ox) - 1, _MAX_VOTES).astype(int)
        ox, oy, ti, dj = ox[pick], oy[pick], ti[pick], dj[pick]

    tol = max(cluster_tol_ratio * float(np.median(np.r_[th, dh])), 4.0)
    offsets = np.stack([ox, oy], axis=1)
    d = np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=2)
    inlier = d <= tol

    best, best_support = -1, 0
    for k in range(len(offsets)):
        sel = inlier[k]
        support = min(len(np.unique(ti[sel])), len(np.unique(dj[sel])))
        if support > best_support:
            best, best_support = k, support
    if best < 0 or best_support < min_support:
        return None

    sel = offsets[inlier[best]]
    return float(np.median(sel[:, 0])), float(np.median(sel[:, 1]))


def _xyah_to_tlbr(xyah):
    x, y, a, h = [float(v) for v in xyah[:4]]
    w = a * h
    return np.array([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dtype=float)


# ====================================================================
# STrack
# ====================================================================
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, flag_fdf=0, detbb=None, landmarks=None):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.flag_fdf = flag_fdf  # class id (0=car, 1=motorcycle in the plate pipeline)
        self.detbb = np.asarray(detbb, dtype=float) if detbb is not None else None
        self.landmarks = np.asarray(landmarks, dtype=float) if landmarks is not None else None

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # lifecycle bookkeeping
        self.hits = 0                 # successful measurement updates
        self.miss_count = 0           # consecutive steps with no match
        self.dt_since_update = 0.0    # elapsed dt (camera frames) since last measurement
        self.recovered = 0            # times rescued by a recovery pass
        self.birth_frame = 0

        # last real observation: recovery from the last-seen box (B2) and ORU (R2)
        self.last_obs_xyah = None
        self.last_obs_state = None
        self.last_obs_frame = 0

    # ---------------- prediction ----------------
    def predict(self, dt=1.0):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, dt)
        self.dt_since_update += float(dt)

    @staticmethod
    def multi_predict(stracks, dt=1.0):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance, dt)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].dt_since_update += float(dt)

    def apply_shift(self, dx, dy):
        if self.mean is not None:
            self.mean[0] += dx
            self.mean[1] += dy

    # ---------------- lifecycle ----------------
    def activate(self, kalman_filter, frame_id, vel_std_scale=1.0):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        xyah = self.tlwh_to_xyah(self._tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate(
            xyah, vel_std_scale=vel_std_scale)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.birth_frame = frame_id
        self.hits = 0
        self.miss_count = 0
        self.dt_since_update = 0.0

        self.last_obs_xyah = np.asarray(xyah, dtype=float).copy()
        self.last_obs_state = (self.mean.copy(), self.covariance.copy())
        self.last_obs_frame = frame_id

    @property
    def last_seen_tlbr(self):
        """Box of the last real detection (not the coasted prediction)."""
        if self.last_obs_xyah is None:
            return self.tlbr
        return _xyah_to_tlbr(self.last_obs_xyah)

    def _seed_velocity(self, new_xyah, max_ratio=1.5):
        """
        FIX 2 / B1: set the velocity of a brand-new track from its first real
        displacement. Called AFTER the Kalman update and it OVERWRITES the
        velocity - v1 added it before the update and the filter then added its
        own correction for the same motion, leaving the track ~1.7x too fast.
        """
        if self.mean is None or self.last_obs_xyah is None:
            return
        elapsed = max(float(self.dt_since_update), 1.0)
        v = (np.asarray(new_xyah, dtype=float) - self.last_obs_xyah) / elapsed

        h = max(float(self.mean[3]), 1.0)
        limit = max_ratio * h
        mag = float(np.hypot(v[0], v[1]))
        if mag > limit and mag > 1e-9:
            v[0] *= limit / mag
            v[1] *= limit / mag
        v[3] = float(np.clip(v[3], -0.25 * h, 0.25 * h))

        self.mean[4] = v[0]
        self.mean[5] = v[1]
        self.mean[6] = 0.0
        self.mean[7] = v[3]

    def _observation_centric_reupdate(self, new_xyah, gap):
        """
        R2 (OC-SORT's ORU): a track that missed `gap - 1` steps has a velocity
        that was only ever extrapolated. Rewind to the last real observation and
        replay the gap with observations interpolated between that box and the
        new one, so the velocity that comes out matches what the car really did.
        """
        mean, cov = self.last_obs_state
        mean, cov = mean.copy(), cov.copy()
        n = int(min(max(gap, 1), 20))
        total_dt = float(self.dt_since_update) if self.dt_since_update > 0 else float(gap)
        step = max(total_dt / n, 1e-3)
        last = self.last_obs_xyah
        new = np.asarray(new_xyah, dtype=float)
        try:
            for k in range(1, n + 1):
                mean, cov = self.kalman_filter.predict(mean, cov, step)
                if k < n:
                    virt = last + (new - last) * (float(k) / n)
                    mean, cov = self.kalman_filter.update(mean, cov, virt)
        except (np.linalg.LinAlgError, ValueError):
            return
        self.mean, self.covariance = mean, cov

    def _measure(self, new_track, frame_id, seed_velocity, seed_max_ratio, oru):
        """Single path for every real measurement (update and re_activate)."""
        new_xyah = self.tlwh_to_xyah(new_track.tlwh)
        gap = int(frame_id - self.last_obs_frame)
        first = bool(seed_velocity) and self.hits == 0 and self.last_obs_xyah is not None

        if oru and not first and gap >= 2 and self.last_obs_state is not None:
            self._observation_centric_reupdate(new_xyah, gap)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_xyah)

        if first:
            self._seed_velocity(new_xyah, seed_max_ratio)

        self.flag_fdf = new_track.flag_fdf
        self.detbb = np.asarray(new_track.detbb, dtype=float) if new_track.detbb is not None else None
        self.landmarks = np.asarray(new_track.landmarks, dtype=float) if new_track.landmarks is not None else None
        self.score = new_track.score

        self.last_obs_xyah = np.asarray(new_xyah, dtype=float).copy()
        self.last_obs_state = (self.mean.copy(), self.covariance.copy())
        self.last_obs_frame = frame_id

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.hits += 1
        self.miss_count = 0
        self.dt_since_update = 0.0

    def re_activate(self, new_track, frame_id, new_id=False,
                    seed_velocity=False, seed_max_ratio=1.5, oru=True, **_unused):
        self._measure(new_track, frame_id, seed_velocity, seed_max_ratio, oru)
        self.tracklet_len = 0
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, seed_velocity=False, seed_max_ratio=1.5,
               oru=True, **_unused):
        self.tracklet_len += 1
        self._measure(new_track, frame_id, seed_velocity, seed_max_ratio, oru)

    # ---------------- geometry ----------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def center(self):
        t = self.tlwh
        return np.array([t[0] + t[2] * 0.5, t[1] + t[3] * 0.5], dtype=float)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=float).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# ====================================================================
# Config
#
# video_processor.py's existing TrackerConfig keeps working untouched - every
# knob below is read with getattr(args, name, default).
# ====================================================================
class PlateTrackerConfig(object):
    def __init__(self,
                 # ---- stock ByteTrack ----
                 track_thresh=0.5,
                 match_thresh=0.99,
                 track_buffer=60,
                 nms_thresh=0.5,
                 mot20=False,
                 second_thresh=0.5,        # low-score association gate
                 duplicate_thresh=0.15,    # remove_duplicate_stracks IoU cost

                 # ---- young-track survival ----
                 predict_unconfirmed=True,
                 unconfirmed_thresh=0.9,
                 unconfirmed_max_miss=5,   # in tracker steps

                 # ---- new-track motion ----
                 new_track_vel_std_scale=3.0,
                 seed_velocity_on_first_update=True,
                 seed_max_ratio=1.5,
                 reseed_after_gap=3.0,     # unused since v2 (ORU replaces it), kept for compat

                 # ---- recovery pass ----
                 recovery_enabled=True,
                 recovery_thresh=0.7,
                 recovery_expansion=0.5,
                 recovery_base_radius=1.5,
                 recovery_radius_growth=0.4,   # per missed tracker step
                 recovery_max_radius=4.0,
                 recovery_shape_weight=0.3,
                 recovery_class_penalty=0.15,
                 recovery_min_size_ratio=0.5,
                 young_recovery_bias=0.05,     # established tracks win ties

                 # ---- low-score (motion blur) recovery ----
                 low_recovery_enabled=True,
                 low_recovery_thresh=0.5,

                 # ---- camera-jolt compensation ----
                 gmc_enabled=True,
                 gmc_min_pairs=2,
                 gmc_max_shift_ratio=0.25,  # of frame height
                 gmc_min_shift_ratio=0.08,  # of median box height; smaller = not a jolt

                 # ---- motion model ----
                 oru_enabled=True,

                 # ---- id continuity ----
                 remove_exited=True,

                 max_removed_history=512):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.nms_thresh = nms_thresh
        self.mot20 = mot20
        self.second_thresh = second_thresh
        self.duplicate_thresh = duplicate_thresh

        self.predict_unconfirmed = predict_unconfirmed
        self.unconfirmed_thresh = unconfirmed_thresh
        self.unconfirmed_max_miss = unconfirmed_max_miss

        self.new_track_vel_std_scale = new_track_vel_std_scale
        self.seed_velocity_on_first_update = seed_velocity_on_first_update
        self.seed_max_ratio = seed_max_ratio
        self.reseed_after_gap = reseed_after_gap

        self.recovery_enabled = recovery_enabled
        self.recovery_thresh = recovery_thresh
        self.recovery_expansion = recovery_expansion
        self.recovery_base_radius = recovery_base_radius
        self.recovery_radius_growth = recovery_radius_growth
        self.recovery_max_radius = recovery_max_radius
        self.recovery_shape_weight = recovery_shape_weight
        self.recovery_class_penalty = recovery_class_penalty
        self.recovery_min_size_ratio = recovery_min_size_ratio
        self.young_recovery_bias = young_recovery_bias

        self.low_recovery_enabled = low_recovery_enabled
        self.low_recovery_thresh = low_recovery_thresh

        self.gmc_enabled = gmc_enabled
        self.gmc_min_pairs = gmc_min_pairs
        self.gmc_max_shift_ratio = gmc_max_shift_ratio
        self.gmc_min_shift_ratio = gmc_min_shift_ratio

        self.oru_enabled = oru_enabled
        self.remove_exited = remove_exited

        self.max_removed_history = max_removed_history


# Backwards-compatible alias
TrackerConfig = PlateTrackerConfig


# ====================================================================
# BYTETracker
# ====================================================================
class BYTETracker(object):
    def __init__(self, args, frame_rate=30, name=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.name = str(name) if name is not None else "?"
        self.log = logging.getLogger("plate_tracker.%s" % self.name)

        g = lambda k, d: getattr(args, k, d)  # noqa: E731

        self.track_thresh = float(g("track_thresh", 0.5))
        self.match_thresh = float(g("match_thresh", 0.99))
        self.track_buffer = int(g("track_buffer", 60))
        self.mot20 = bool(g("mot20", False))
        self.second_thresh = float(g("second_thresh", 0.5))
        self.duplicate_thresh = float(g("duplicate_thresh", 0.15))

        self.predict_unconfirmed = bool(g("predict_unconfirmed", True))
        self.unconfirmed_thresh = float(g("unconfirmed_thresh", 0.9))
        self.unconfirmed_max_miss = int(g("unconfirmed_max_miss", 5))

        self.new_track_vel_std_scale = float(g("new_track_vel_std_scale", 3.0))
        self.seed_velocity = bool(g("seed_velocity_on_first_update", True))
        self.seed_max_ratio = float(g("seed_max_ratio", 1.5))

        self.recovery_enabled = bool(g("recovery_enabled", True))
        self.recovery_thresh = float(g("recovery_thresh", 0.7))
        self.recovery_expansion = float(g("recovery_expansion", 0.5))
        self.recovery_base_radius = float(g("recovery_base_radius", 1.5))
        self.recovery_radius_growth = float(g("recovery_radius_growth", 0.4))
        self.recovery_max_radius = float(g("recovery_max_radius", 4.0))
        self.recovery_shape_weight = float(g("recovery_shape_weight", 0.3))
        self.recovery_class_penalty = float(g("recovery_class_penalty", 0.15))
        self.recovery_min_size_ratio = float(g("recovery_min_size_ratio", 0.5))
        self.young_recovery_bias = float(g("young_recovery_bias", 0.05))

        self.low_recovery_enabled = bool(g("low_recovery_enabled", True))
        self.low_recovery_thresh = float(g("low_recovery_thresh", 0.5))

        self.gmc_enabled = bool(g("gmc_enabled", True))
        self.gmc_min_pairs = int(g("gmc_min_pairs", 2))
        self.gmc_max_shift_ratio = float(g("gmc_max_shift_ratio", 0.25))
        self.gmc_min_shift_ratio = float(g("gmc_min_shift_ratio", 0.08))

        self.oru_enabled = bool(g("oru_enabled", True))
        self.remove_exited = bool(g("remove_exited", True))

        self.removed_stracks = deque(maxlen=int(g("max_removed_history", 512)))

        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.stats = {
            "created": 0,
            "confirmed": 0,
            "lost": 0,
            "refind_iou": 0,
            "refind_low": 0,
            "refind_recovery": 0,
            "removed_unconfirmed": 0,
            "removed_timeout": 0,
            "removed_exited": 0,
            "gmc_applied": 0,
        }

    # ---------------- public helpers ----------------
    def get_stats(self):
        s = dict(self.stats)
        s["frame_id"] = self.frame_id
        s["active"] = len([t for t in self.tracked_stracks if t.is_activated])
        s["unconfirmed"] = len([t for t in self.tracked_stracks if not t.is_activated])
        s["lost_now"] = len(self.lost_stracks)
        return s

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks.clear()
        self.frame_id = 0
        for k in self.stats:
            self.stats[k] = 0

    # ---------------- internals ----------------
    def _kw(self):
        return dict(seed_velocity=self.seed_velocity,
                    seed_max_ratio=self.seed_max_ratio,
                    oru=self.oru_enabled)

    def _recovery_cost(self, tracks, pred_boxes, cand_boxes, cand_classes, shift):
        """min(recovery cost from predicted box, from last-seen box)  (B2)."""
        pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4).copy()
        last = np.array([t.last_seen_tlbr for t in tracks], dtype=float).reshape(-1, 4)
        if shift is not None:
            pred[:, [0, 2]] += shift[0]
            pred[:, [1, 3]] += shift[1]
            last[:, [0, 2]] += shift[0]
            last[:, [1, 3]] += shift[1]
        missing = [max(self.frame_id - t.last_obs_frame, 1) for t in tracks]
        classes = [int(round(float(t.flag_fdf))) for t in tracks]
        kw = dict(track_classes=classes, det_classes=cand_classes,
                  expansion=self.recovery_expansion,
                  base_radius=self.recovery_base_radius,
                  radius_growth=self.recovery_radius_growth,
                  max_radius=self.recovery_max_radius,
                  shape_weight=self.recovery_shape_weight,
                  class_penalty=self.recovery_class_penalty,
                  min_size_ratio=self.recovery_min_size_ratio)
        c1 = recovery_distance(pred, cand_boxes, missing, **kw)
        c2 = recovery_distance(last, cand_boxes, missing, **kw)
        return np.minimum(c1, c2)

    def _exited(self, track, img_w, img_h, margin=0.02, min_visible=0.5):
        """
        R3: the car was last seen cut off by a frame border and is not coming
        back: either it was moving out through that border and has now been
        missing for 2+ steps, or its prediction is already mostly off-frame.
        Without this, the lost track sits at the edge with a growing gate for
        the whole track_buffer and hands its id to the next car arriving there.
        """
        lb = track.last_seen_tlbr
        mx, my = margin * img_w, margin * img_h
        at_l, at_t = lb[0] <= mx, lb[1] <= my
        at_r, at_b = lb[2] >= img_w - mx, lb[3] >= img_h - my
        if not (at_l or at_t or at_r or at_b):
            return False
        pb = track.tlbr
        area = max((pb[2] - pb[0]) * (pb[3] - pb[1]), 1e-6)
        iw = max(0.0, min(pb[2], img_w) - max(pb[0], 0.0))
        ih = max(0.0, min(pb[3], img_h) - max(pb[1], 0.0))
        if (iw * ih) / area < min_visible:
            return True
        vx, vy = float(track.mean[4]), float(track.mean[5])
        outward = (at_l and vx < 0) or (at_r and vx > 0) or \
                  (at_t and vy < 0) or (at_b and vy > 0)
        return outward and (self.frame_id - track.last_obs_frame) >= 2

    # ---------------- main entry point ----------------
    def update(self, output_results, img_info, img_size, dt=None):
        """
        output_results : Nx6 array [x1, y1, x2, y2, score, class_id]
        img_info       : (h, w) of the frame the boxes came from
        img_size       : (h, w) the boxes should be scaled to
        dt             : camera frames elapsed since the previous update
                         (1 + skipped frames). Defaults to 1.
        """
        self.frame_id += 1
        fid = self.frame_id
        step_dt = 1.0 if dt is None else max(float(dt), 1e-3)
        kw = self._kw()

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # ---------------- parse detections ----------------
        if output_results is None or len(output_results) == 0:
            output_results = np.zeros((0, 6), dtype=np.float64)
        output_results = np.asarray(output_results, dtype=np.float64).copy()
        if output_results.ndim == 1:
            output_results = (output_results.reshape(1, -1)
                              if output_results.size >= 5
                              else np.zeros((0, 6), dtype=np.float64))
        if output_results.shape[1] < 5:
            raise ValueError(f"Unexpected detection shape: {output_results.shape}")

        img_h, img_w = float(img_info[0]), float(img_info[1])
        scale = min(img_size[0] / img_h, img_size[1] / img_w)

        bboxes = output_results[:, :4] / scale
        scores = output_results[:, 4]
        flags = output_results[:, 5] if output_results.shape[1] >= 6 else np.zeros_like(scores)
        landmarks = output_results[:, 6:] if output_results.shape[1] > 6 else None

        # finite, non-degenerate boxes only
        valid = np.isfinite(bboxes).all(axis=1) & np.isfinite(scores) & \
            (bboxes[:, 2] - bboxes[:, 0] > 1.0) & (bboxes[:, 3] - bboxes[:, 1] > 1.0)

        remain_inds = valid & (scores > self.track_thresh)
        inds_second = valid & (scores > 0.1) & (scores <= self.track_thresh)   # B5

        def _make(mask):
            out = []
            for i in np.nonzero(mask)[0]:
                tlbr = bboxes[i]
                lm = landmarks[i] if landmarks is not None else None
                out.append(STrack(STrack.tlbr_to_tlwh(tlbr), float(scores[i]), flags[i],
                                  detbb=tlbr, landmarks=lm))
            return out

        detections = _make(remain_inds)
        detections_second = _make(inds_second)

        # ---------------- split confirmed / unconfirmed ----------------
        unconfirmed = []
        tracked = []
        for track in self.tracked_stracks:
            (tracked if track.is_activated else unconfirmed).append(track)

        # ================================================================
        # Step 1: predict (FIX 1: unconfirmed tracks too)
        # ================================================================
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        if self.predict_unconfirmed and unconfirmed:
            STrack.multi_predict(strack_pool + unconfirmed, step_dt)
        else:
            STrack.multi_predict(strack_pool, step_dt)

        pool_boxes = np.array([t.tlbr for t in strack_pool], dtype=float).reshape(-1, 4)
        unc_boxes = np.array([t.tlbr for t in unconfirmed], dtype=float).reshape(-1, 4)
        det_boxes = np.array([d.tlbr for d in detections], dtype=float).reshape(-1, 4)
        det_scores = np.array([d.score for d in detections], dtype=float)
        det_classes = np.array([int(round(float(d.flag_fdf))) for d in detections], dtype=int)
        sec_boxes = np.array([d.tlbr for d in detections_second], dtype=float).reshape(-1, 4)
        sec_classes = np.array([int(round(float(d.flag_fdf))) for d in detections_second],
                               dtype=int)

        # only well-established tracks say anything about camera motion
        reliable = [t.state == TrackState.Tracked and t.hits >= 3 for t in strack_pool]
        gmc_residuals = []

        def _centre(box):
            return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=float)

        def _apply_pool(track, det, via):
            if track.state == TrackState.Tracked:
                track.update(det, fid, **kw)
                activated_stracks.append(track)
            else:
                missed = fid - track.last_obs_frame - 1
                track.re_activate(det, fid, new_id=False, **kw)
                refind_stracks.append(track)
                self.log.debug("[TRK-REFIND] id=%s via=%s missed=%d steps conf=%.2f",
                               track.track_id, via, missed, float(det.score))
                return True
            return False

        def _apply_young(track, det, via):
            track.update(det, fid, **kw)
            activated_stracks.append(track)
            self.stats["confirmed"] += 1
            self.log.debug("[TRK-CONFIRM] id=%s via=%s after=%df", track.track_id, via,
                           fid - track.birth_frame)

        # ================================================================
        # Step 2: high-score detections vs confirmed + lost, IoU * score
        # ================================================================
        dists = iou_distance_boxes(pool_boxes, det_boxes)
        if not self.mot20:
            dists = fuse_score_array(dists, det_scores)
        matches, u_pool, u_det = linear_assignment(dists, thresh=self.match_thresh)
        for ip, idd in matches:
            ip, idd = int(ip), int(idd)
            if reliable[ip]:
                gmc_residuals.append(_centre(det_boxes[idd]) - _centre(pool_boxes[ip]))
            if _apply_pool(strack_pool[ip], detections[idd], "iou"):
                self.stats["refind_iou"] += 1
        u_pool = [int(i) for i in u_pool]
        u_det = [int(i) for i in u_det]

        # ================================================================
        # Step 2b (B6): young tracks get their plain IoU match BEFORE any
        # wide-gate recovery can hand their detection to someone else
        # ================================================================
        u_unc = list(range(len(unconfirmed)))
        if unconfirmed and u_det:
            d = iou_distance_boxes(unc_boxes, det_boxes[u_det])
            if not self.mot20:
                d = fuse_score_array(d, det_scores[u_det])
            m, uu, ud = linear_assignment(d, thresh=self.unconfirmed_thresh)
            for iu, idd in m:
                _apply_young(unconfirmed[int(iu)], detections[u_det[int(idd)]], "iou")
            u_unc = [int(i) for i in uu]
            u_det = [u_det[int(i)] for i in ud]

        # ================================================================
        # Step 3: low-score detections vs still-tracked + young (plain IoU)
        # R1: young tracks are included - on a bump the blurred car often
        # scores under track_thresh, which used to be invisible to them.
        # ================================================================
        u_sec = list(range(len(detections_second)))
        rows = [("p", i) for i in u_pool if strack_pool[i].state == TrackState.Tracked] + \
               [("u", i) for i in u_unc]
        if rows and u_sec:
            rb = np.array([pool_boxes[i] if k == "p" else unc_boxes[i] for k, i in rows],
                          dtype=float).reshape(-1, 4)
            d = iou_distance_boxes(rb, sec_boxes)
            m, _, us = linear_assignment(d, thresh=self.second_thresh)
            taken_p, taken_u = set(), set()
            for ir, idd in m:
                kind, i = rows[int(ir)]
                det = detections_second[int(idd)]
                if kind == "p":
                    if reliable[i]:
                        gmc_residuals.append(_centre(sec_boxes[int(idd)]) - _centre(pool_boxes[i]))
                    _apply_pool(strack_pool[i], det, "low")
                    taken_p.add(i)
                else:
                    _apply_young(unconfirmed[i], det, "low")
                    taken_u.add(i)
                self.stats["refind_low"] += 1
            u_pool = [i for i in u_pool if i not in taken_p]
            u_unc = [i for i in u_unc if i not in taken_u]
            u_sec = [int(i) for i in us]

        # ================================================================
        # Step 3.5: camera-jolt estimate (FIX 6, B3)
        # ================================================================
        shift = None
        if self.gmc_enabled:
            max_shift = self.gmc_max_shift_ratio * img_h
            hs = [b[3] - b[1] for b in pool_boxes] + [b[3] - b[1] for b in det_boxes]
            min_shift = max(4.0, self.gmc_min_shift_ratio * (float(np.median(hs)) if hs else 0.0))
            shift = estimate_global_shift(gmc_residuals, min_pairs=self.gmc_min_pairs,
                                          max_shift=max_shift, min_shift=min_shift)
            source = "matched"
            if shift is None and u_det and (u_pool or u_unc):
                vote_boxes = np.vstack([pool_boxes[u_pool].reshape(-1, 4),
                                        unc_boxes[u_unc].reshape(-1, 4)])
                shift = vote_global_shift(vote_boxes, det_boxes[u_det],
                                          min_support=self.gmc_min_pairs,
                                          max_shift=max_shift)
                source = "vote"
            if shift is not None and float(np.hypot(*shift)) >= min_shift:
                self.stats["gmc_applied"] += 1
                self.log.debug("[GMC] shift=(%.1f, %.1f) via=%s", shift[0], shift[1], source)
            else:
                shift = None

        # ================================================================
        # Step 4: joint RECOVERY pass (FIX 5, B2, B6) - no overlap needed.
        # Every leftover track (tracked, lost, young) against every leftover
        # high-score detection, from both its predicted and last-seen box.
        # ================================================================
        if self.recovery_enabled and (u_pool or u_unc) and u_det:
            rows = [("p", i) for i in u_pool] + [("u", i) for i in u_unc]
            rtracks = [strack_pool[i] if k == "p" else unconfirmed[i] for k, i in rows]
            rpred = np.array([pool_boxes[i] if k == "p" else unc_boxes[i] for k, i in rows],
                             dtype=float).reshape(-1, 4)
            rcost = self._recovery_cost(rtracks, rpred, det_boxes[u_det],
                                        det_classes[u_det], shift)
            if self.young_recovery_bias > 0:
                young_rows = np.array([k == "u" for k, _ in rows])
                ok = rcost < _REJECT
                rcost[young_rows[:, None] & ok] += self.young_recovery_bias

            m, ur, ud = linear_assignment(rcost, thresh=self.recovery_thresh)
            taken_p, taken_u = set(), set()
            for ir, idc in m:
                kind, i = rows[int(ir)]
                track = rtracks[int(ir)]
                det = detections[u_det[int(idc)]]
                cost = float(rcost[int(ir), int(idc)])
                missed = fid - track.last_obs_frame - 1
                track.recovered += 1
                self.stats["refind_recovery"] += 1
                if kind == "p":
                    _apply_pool(track, det, "recovery")
                    taken_p.add(i)
                    via = "recovery"
                else:
                    _apply_young(track, det, "recovery")
                    taken_u.add(i)
                    via = "recovery-young"
                self.log.info("[TRK-REFIND] id=%s via=%s cost=%.2f missed=%d steps conf=%.2f%s",
                              track.track_id, via, cost, missed, float(det.score),
                              "" if shift is None else " gmc=(%.0f,%.0f)" % shift)
            u_pool = [i for i in u_pool if i not in taken_p]
            u_unc = [i for i in u_unc if i not in taken_u]
            u_det = [u_det[int(i)] for i in ud]

        # ================================================================
        # Step 4b (R1): strict low-score recovery for cars that were being
        # tracked a moment ago - blurred AND jumped, the classic bump frame.
        # Lost tracks are excluded: a weak detection is not enough to revive them.
        # ================================================================
        if self.recovery_enabled and self.low_recovery_enabled and u_sec:
            rows = [("p", i) for i in u_pool if strack_pool[i].state == TrackState.Tracked] + \
                   [("u", i) for i in u_unc if unconfirmed[i].hits == 0 and
                    fid - unconfirmed[i].last_obs_frame <= 1]
            if rows:
                rtracks = [strack_pool[i] if k == "p" else unconfirmed[i] for k, i in rows]
                rpred = np.array([pool_boxes[i] if k == "p" else unc_boxes[i]
                                  for k, i in rows], dtype=float).reshape(-1, 4)
                rcost = self._recovery_cost(rtracks, rpred, sec_boxes[u_sec],
                                            sec_classes[u_sec], shift)
                m, _, us = linear_assignment(rcost, thresh=self.low_recovery_thresh)
                taken_p, taken_u = set(), set()
                for ir, idc in m:
                    kind, i = rows[int(ir)]
                    track = rtracks[int(ir)]
                    det = detections_second[u_sec[int(idc)]]
                    track.recovered += 1
                    self.stats["refind_recovery"] += 1
                    if kind == "p":
                        _apply_pool(track, det, "recovery-low")
                        taken_p.add(i)
                    else:
                        _apply_young(track, det, "recovery-low")
                        taken_u.add(i)
                    self.log.info("[TRK-REFIND] id=%s via=recovery-low cost=%.2f conf=%.2f",
                                  track.track_id, float(rcost[int(ir), int(idc)]),
                                  float(det.score))
                u_pool = [i for i in u_pool if i not in taken_p]
                u_unc = [i for i in u_unc if i not in taken_u]
                u_sec = [u_sec[int(i)] for i in us]

        # ---- confirmed tracks that found nothing become lost ----
        for i in u_pool:
            track = strack_pool[i]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                track.miss_count += 1
                lost_stracks.append(track)
                self.stats["lost"] += 1
                self.log.debug("[TRK-LOST] id=%s age=%df hits=%d - no detection matched",
                               track.track_id, fid - track.birth_frame, track.hits)
            else:
                track.miss_count += 1

        # ---- young tracks that matched nothing: grace period, not death ----
        for i in u_unc:
            track = unconfirmed[i]
            track.miss_count += 1
            if track.miss_count > self.unconfirmed_max_miss:
                track.mark_removed()
                removed_stracks.append(track)
                self.stats["removed_unconfirmed"] += 1
                self.log.info(
                    "[TRK-REMOVE] id=%s reason=unconfirmed_expired age=%df hits=%d misses=%d",
                    track.track_id, fid - track.birth_frame, track.hits, track.miss_count)
            else:
                self.log.debug("[TRK-COAST] id=%s unconfirmed miss=%d/%d age=%df",
                               track.track_id, track.miss_count, self.unconfirmed_max_miss,
                               fid - track.birth_frame)

        # ================================================================
        # Step 5: brand new tracks from what is left
        # ================================================================
        for idx in u_det:
            track = detections[idx]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, fid,
                           vel_std_scale=self.new_track_vel_std_scale)
            activated_stracks.append(track)
            self.stats["created"] += 1
            self.log.debug("[TRK-NEW] id=%s fid=%d cls=%d conf=%.2f box=(%d,%d,%d,%d)",
                           track.track_id, fid, int(round(float(track.flag_fdf))),
                           float(track.score), *[int(v) for v in track.tlbr])

        # ================================================================
        # Step 6: retire lost tracks that timed out or left the frame (R3)
        # ================================================================
        for track in joint_stracks(self.lost_stracks, lost_stracks):
            if track.state != TrackState.Lost:
                continue
            reason = None
            if fid - track.end_frame > self.max_time_lost:
                reason = "lost_timeout"
                self.stats["removed_timeout"] += 1
            elif self.remove_exited and self._exited(track, img_w, img_h):
                reason = "exited"
                self.stats["removed_exited"] += 1
            if reason:
                track.mark_removed()
                removed_stracks.append(track)
                self.log.debug("[TRK-REMOVE] id=%s reason=%s hits=%d recovered=%d",
                               track.track_id, reason, track.hits, track.recovered)

        # ================================================================
        # Step 7: merge the bookkeeping lists (state is the source of truth)
        # ================================================================
        tracked_all = joint_stracks(self.tracked_stracks, activated_stracks)
        tracked_all = joint_stracks(tracked_all, refind_stracks)
        self.tracked_stracks = [t for t in tracked_all if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in joint_stracks(self.lost_stracks, lost_stracks)
                             if t.state == TrackState.Lost]
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, self.duplicate_thresh)

        return [t for t in self.tracked_stracks if t.is_activated]


# ====================================================================
# List helpers
# ====================================================================
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, thresh=0.15):
    if not stracksa or not stracksb:
        return stracksa, stracksb
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < thresh)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
