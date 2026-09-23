"""
plate_tracker.py  -  ByteTrack for the plate system  ("plates are not forgotten" edition)
====================================================================================

WHY TRACKS WERE BEING LOST  (read this before touching the knobs)
------------------------------------------------------------------------------------
Symptom you reported: a car is detected with good confidence (s0.86), the
overlay says `dets 1 | tracks 0(+1 ghost)`, the track is `age 2f`, and the
pipeline then logs `DROPPED: seen 7<8`. The plate is forgotten even though
YOLO never stopped seeing the car.

Stock ByteTrack has a two-tier track lifecycle:

    frame N   : detection with no match  ->  activate() a NEW track
                but `is_activated = False`  (it is "unconfirmed")
    frame N+1 : the unconfirmed track must match a detection, or it is
                `mark_removed()` - killed outright, no second chance.

Three things in the stock implementation conspire to kill those unconfirmed
tracks on exactly the frames where a plate matters most:

  (1) UNCONFIRMED TRACKS WERE NEVER PREDICTED.
      `STrack.multi_predict(strack_pool)` is called on tracked + lost tracks.
      `unconfirmed` is NOT in that pool, so a one-frame-old track is compared
      against the next frame using its ORIGINAL box, with no motion applied at
      all. At 30 fps a car barely moves in one frame and nobody notices. Your
      engine loop runs at ~9 fps on a ~25-30 fps stream, so every tracker step
      is 3 camera frames of real motion - and the comparison box is stale by
      all 3 of them.

  (2) A BRAND NEW TRACK HAS VELOCITY ZERO.
      `KalmanFilter.initiate()` sets `mean_vel = zeros`. So even once (1) is
      fixed, the very first prediction says "the car did not move", which is
      the worst possible guess for a moving vehicle at a 3-frame step.

  (3) THE UNCONFIRMED GATE IS THE STRICTEST GATE IN THE FILE, AND IT IS FATAL.
      `linear_assignment(dists, thresh=0.7)` on a fuse_score cost means a match
      needs `iou * det_score >= 0.30`; at score 0.86 that is `iou >= 0.35`.
      Miss it once and the track is removed permanently. The next frame the
      same car starts a fresh track id, which then faces the same gauntlet.
      A car can cycle through this repeatedly and never accumulate the 8
      `seen_frames` that `_finalize_or_drop_track()` requires - hence
      `DROPPED: seen 7<8` on a car that was visible the whole time.

A bump/jolt is simply the worst case of the same mechanism: the box jumps
further than usual in one step, IoU collapses, and a young track dies.

WHAT THIS VERSION CHANGES
------------------------------------------------------------------------------------
  FIX 1  Unconfirmed tracks are predicted with the Kalman filter like every
         other track.                                     -> predict_unconfirmed
  FIX 2  Brand-new tracks seed their velocity from the first observed
         displacement instead of starting at zero, and start with a wider
         velocity covariance.        -> seed_velocity_on_first_update / vel_std_scale
  FIX 3  An unconfirmed track that misses gets a grace period instead of being
         deleted, and its gate is loosened.   -> unconfirmed_max_miss / unconfirmed_thresh
  FIX 4  Real elapsed time per step. `update(..., dt=N)` tells the filter how
         many camera frames actually passed, so skipped frames stop corrupting
         the motion model.                                              -> dt
  FIX 5  A recovery association pass that does not need IoU overlap at all:
         expanded-box IoU + centre distance normalised by box height + shape
         similarity, with a gate that widens the longer a track has been
         missing. This is what rescues the "box jumped, zero overlap, same car"
         case.                                                 -> recovery_*
  FIX 6  Camera-jolt compensation (GMC-lite): the median residual of the tracks
         that DID match estimates a global image shift, which is applied to the
         unmatched tracks before the recovery pass. When the camera shakes,
         every box moves together, and the matched boxes reveal by how much.
                                                                        -> gmc_*
  FIX 7  `removed_stracks` no longer grows without bound (stock ByteTrack leaks
         it forever and re-scans it every frame).           -> max_removed_history
  FIX 8  Logging and counters, in the same tag style as video_processor.py:
         [TRK-NEW] [TRK-CONFIRM] [TRK-LOST] [TRK-REFIND] [TRK-REMOVE] [GMC].
         `[TRK-REMOVE] reason=unconfirmed_expired` is literally a forgotten
         plate - if you still see those, raise unconfirmed_max_miss.
         `get_stats()` returns the counters for a periodic summary line.

TO GO BACK TO STOCK BYTETRACK BEHAVIOUR (for an A/B comparison) set:
    predict_unconfirmed=False, seed_velocity_on_first_update=False,
    new_track_vel_std_scale=1.0, unconfirmed_thresh=0.7, unconfirmed_max_miss=0,
    recovery_enabled=False, gmc_enabled=False
...and call update() without dt.

INTERFACE  (unchanged - drop-in for yolox.tracker.byte_tracker)
------------------------------------------------------------------------------------
    BYTETracker(args, frame_rate=30, name=None)
        args needs .track_thresh .match_thresh .track_buffer .mot20
        (the existing TrackerConfig in video_processor.py already provides
        these; every new knob is read with getattr() and has a default, so
        nothing in video_processor.py has to change)
    tracker.update(output_results, img_info, img_size, dt=None) -> list[STrack]
        output_results: Nx6  [x1, y1, x2, y2, score, class_id]
        dt: optional, how many camera frames elapsed since the last update
    each returned STrack exposes .track_id .score .flag_fdf .detbb .tlwh .tlbr
"""

import logging
from collections import OrderedDict, deque

import numpy as np
import scipy.linalg
import lap

try:
    from cython_bbox import bbox_overlaps as _bbox_ious_cython
except Exception:  # pragma: no cover - fallback keeps the file usable anywhere
    _bbox_ious_cython = None


logger = logging.getLogger("plate_tracker")

# Cost value used for pairs that a hard gate rejected. Must stay far above any
# association threshold so lap.lapjv can never pick it.
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
# Kalman filter
#
# Same 8-state constant-velocity model as before (x, y, a, h + velocities).
# Two changes vs the reference implementation:
#   * predict()/multi_predict() take a real `dt` (in tracker steps) so that
#     skipped camera frames advance the motion model correctly, and the
#     process noise grows with dt instead of pretending every step is equal.
#   * initiate() can widen the initial velocity uncertainty, so a brand new
#     track does not fight its first real measurement.
# ====================================================================
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.
        self._ndim = ndim

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # motion matrices for non-unit dt, built on demand and cached
        self._motion_mat_cache = {1.0: self._motion_mat}

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def motion_mat(self, dt=1.0):
        """Constant-velocity transition matrix for an arbitrary step size."""
        dt = float(dt)
        mat = self._motion_mat_cache.get(dt)
        if mat is None:
            ndim = self._ndim
            mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                mat[i, ndim + i] = dt
            self._motion_mat_cache[dt] = mat
        return mat

    def initiate(self, measurement, vel_std_scale=1.0):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        vel_std_scale : float
            Multiplier on the initial velocity uncertainty. > 1 tells the
            filter "I have no idea how fast this thing is going", which lets
            the first measurement move the velocity estimate much further.
            (FIX 2)

        Returns
        -------
        (ndarray, ndarray)
            Mean vector (8 dim) and covariance matrix (8x8) of the new track.
        """
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
        """Run Kalman filter prediction step (single track, arbitrary dt)."""
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
        # process noise accumulates with elapsed time (FIX 4)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * dt

        mm = self.motion_mat(dt)
        mean = np.dot(mean, mm.T)
        covariance = np.linalg.multi_dot((mm, covariance, mm.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
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
        """Run Kalman filter prediction step (vectorized, arbitrary dt)."""
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

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mm = self.motion_mat(dt)
        mean = np.dot(mean, mm.T)
        left = np.dot(mm, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, mm.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
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
        """Squared Mahalanobis distance between a state and N measurements."""
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
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


# ====================================================================
# Assignment / cost helpers
# ====================================================================
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def fuse_score(cost_matrix, detections):
    """Fold detection confidence into the IoU cost (stock ByteTrack)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def _ious_numpy(atlbrs, btlbrs):
    """Vectorized IoU fallback when cython_bbox is unavailable."""
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
    """
    Compute IoU matrix.
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if out.size == 0:
        return out

    if _bbox_ious_cython is not None:
        return _bbox_ious_cython(
            np.ascontiguousarray(atlbrs, dtype=float),
            np.ascontiguousarray(btlbrs, dtype=float)
        )
    return _ious_numpy(atlbrs, btlbrs)


def iou_distance(atracks, btracks):
    """
    Cost = 1 - IoU. Accepts either STrack lists or raw Nx4 tlbr arrays.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


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
    """
    Grow each box by `ratio` of its own size (half on each side).
    Used by the recovery pass so that boxes which *nearly* overlap still
    produce a usable IoU signal. (FIX 5)
    """
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
                      class_penalty=0.15):
    """
    Overlap-free association cost, in [0, 1], or _REJECT for gated-out pairs.

    Built from three signals, none of which require the boxes to overlap:

      proximity : centre distance normalised by the mean box height, divided by
                  a gate radius that GROWS with how long the track has been
                  missing (an object unseen for 5 steps is allowed to be
                  further away than one unseen for 1).
      shape     : how similar the two boxes are in width and height - the same
                  car after a jolt keeps its size, a different object usually
                  does not.
      expanded  : IoU of the boxes after inflating both by `expansion`. If the
      IoU         inflated boxes overlap that is strong evidence, and it wins
                  over the proximity term.

    class_penalty adds a soft cost (not a hard veto) when the detector class
    disagrees, so a car track does not get rescued onto a motorcycle.
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

    # --- proximity, in units of "mean box heights" -------------------
    dist = np.sqrt((tcx[:, None] - dcx[None, :]) ** 2 +
                   (tcy[:, None] - dcy[None, :]) ** 2)
    scale = np.maximum(0.5 * (th[:, None] + dh[None, :]), 1.0)
    ndist = dist / scale

    missing = np.asarray(frames_missing, dtype=float).reshape(-1, 1)
    radius = np.minimum(base_radius + radius_growth * missing, max_radius)
    radius = np.maximum(radius, 1e-6)
    prox_cost = np.clip(ndist / radius, 0.0, 1.0)

    # --- shape similarity --------------------------------------------
    w_sim = np.minimum(tw[:, None], dw[None, :]) / np.maximum(tw[:, None], dw[None, :])
    h_sim = np.minimum(th[:, None], dh[None, :]) / np.maximum(th[:, None], dh[None, :])
    shape_cost = 1.0 - (w_sim * h_sim)

    cost = (1.0 - shape_weight) * prox_cost + shape_weight * shape_cost

    # --- expanded IoU overrides proximity when the boxes do overlap ---
    eiou = ious(expand_boxes(tb, expansion), expand_boxes(db, expansion))
    eiou = np.asarray(eiou, dtype=float).reshape(n, m)
    cost = np.minimum(cost, 1.0 - eiou)

    # --- soft class disagreement penalty ------------------------------
    if class_penalty > 0 and track_classes is not None and det_classes is not None:
        tc = np.asarray(track_classes).reshape(-1, 1)
        dc = np.asarray(det_classes).reshape(1, -1)
        cost = cost + class_penalty * (tc != dc).astype(float)

    cost = np.clip(cost, 0.0, 1.0)

    # --- hard gate: too far AND no expanded overlap -> not a candidate -
    gate = (ndist > radius) & (eiou <= 0.0)
    cost[gate] = _REJECT
    return cost


def estimate_global_shift(residuals, min_pairs=2, max_shift=None):
    """
    GMC-lite, evidence from tracks that DID match (FIX 6, part 1).

    `residuals` are (detection_centre - predicted_track_centre) vectors for the
    tracks that matched this frame. If the camera jolted, every box in the
    image moved by roughly the same amount, so the median residual is a cheap,
    pixel-free estimate of that global shift - no optical flow, no frame data.

    Returns (dx, dy), or None when there is not enough evidence or the estimate
    is implausibly large (better to give up than to teleport tracks).
    """
    if residuals is None or len(residuals) < max(1, int(min_pairs)):
        return None
    r = np.asarray(residuals, dtype=float).reshape(-1, 2)
    dx = float(np.median(r[:, 0]))
    dy = float(np.median(r[:, 1]))
    if max_shift is not None and float(np.hypot(dx, dy)) > max_shift:
        return None
    return dx, dy


def vote_global_shift(track_boxes, det_boxes, min_support=2, max_shift=None,
                      size_tol=0.45, cluster_tol_ratio=0.6):
    """
    GMC-lite, evidence from tracks that did NOT match (FIX 6, part 2).

    The matched-residual estimate above has a chicken-and-egg problem: a jolt
    big enough to matter is a jolt big enough that NOTHING matches, so there
    are no residuals to take a median of.

    So vote instead. Every plausible (unmatched track, unmatched detection)
    pairing proposes the offset that would align it. A real camera shift moves
    every object by the same vector, so the correct offset collects one vote
    per object while coincidental pairings scatter. The offset with the most
    support wins.

    Requires at least `min_support` independent objects to agree - with a
    single object in frame you genuinely cannot tell "the camera moved" from
    "the car moved", so this returns None rather than guessing.
    """
    tb = np.asarray(track_boxes, dtype=float).reshape(-1, 4)
    db = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    n, m = len(tb), len(db)
    min_support = max(2, int(min_support))
    if n == 0 or m == 0 or n * m < min_support:
        return None

    tcx = (tb[:, 0] + tb[:, 2]) * 0.5
    tcy = (tb[:, 1] + tb[:, 3]) * 0.5
    tw = np.maximum(tb[:, 2] - tb[:, 0], 1.0)
    th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
    dcx = (db[:, 0] + db[:, 2]) * 0.5
    dcy = (db[:, 1] + db[:, 3]) * 0.5
    dw = np.maximum(db[:, 2] - db[:, 0], 1.0)
    dh = np.maximum(db[:, 3] - db[:, 1], 1.0)

    # only pair boxes that could plausibly be the same object
    w_sim = np.minimum(tw[:, None], dw[None, :]) / np.maximum(tw[:, None], dw[None, :])
    h_sim = np.minimum(th[:, None], dh[None, :]) / np.maximum(th[:, None], dh[None, :])
    ok = (w_sim >= size_tol) & (h_sim >= size_tol)
    if not ok.any():
        return None

    ox = (dcx[None, :] - tcx[:, None])[ok]
    oy = (dcy[None, :] - tcy[:, None])[ok]
    if max_shift is not None:
        keep = np.hypot(ox, oy) <= max_shift
        ox, oy = ox[keep], oy[keep]
    if len(ox) < min_support:
        return None
    # keep the vote cheap in busy scenes (the clustering below is O(k^2))
    _MAX_VOTES = 600
    if len(ox) > _MAX_VOTES:
        pick = np.linspace(0, len(ox) - 1, _MAX_VOTES).astype(int)
        ox, oy = ox[pick], oy[pick]

    tol = max(cluster_tol_ratio * float(np.median(np.r_[th, dh])), 4.0)
    offsets = np.stack([ox, oy], axis=1)
    d = np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=2)
    inlier = d <= tol
    support = inlier.sum(axis=1)
    best = int(np.argmax(support))
    if int(support[best]) < min_support:
        return None

    sel = offsets[inlier[best]]
    return float(np.median(sel[:, 0])), float(np.median(sel[:, 1]))


# ====================================================================
# STrack
# ====================================================================
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, flag_fdf=0, detbb=None, landmarks=None):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.flag_fdf = flag_fdf  # class id (0=car, 1=motorcycle in the plate pipeline)
        self.detbb = np.asarray(detbb, dtype=float) if detbb is not None else None  # original xyxy, used for cropping
        self.landmarks = np.asarray(landmarks, dtype=float) if landmarks is not None else None  # unused by plates, kept for parity

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # --- lifecycle bookkeeping used by the new logic / logging ---
        self.hits = 0                 # number of successful measurement updates
        self.miss_count = 0           # consecutive updates with no match
        self.dt_since_update = 0.0    # elapsed steps since the last measurement
        self.recovered = 0            # times rescued by the recovery pass
        self.birth_frame = 0

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
        """Shift the predicted centre (used only on copies for GMC scoring)."""
        if self.mean is not None:
            self.mean[0] += dx
            self.mean[1] += dy

    # ---------------- lifecycle ----------------
    def activate(self, kalman_filter, frame_id, vel_std_scale=1.0):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh), vel_std_scale=vel_std_scale)

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

    def _seed_velocity(self, new_xyah, max_ratio=1.5):
        """
        FIX 2: push the observed displacement straight into the velocity state.

        A track created last step has vx = vy = 0, so its prediction says the
        car stood still - at 9 fps on a 30 fps stream that is 3 frames of real
        motion thrown away, and it is the single biggest reason young tracks
        fail to match. The residual between where we predicted the box and
        where the detector actually found it IS the velocity, so we add it in
        (for a fresh track the prediction equals the old position, so this is
        exactly the measured displacement).

        Clamped to `max_ratio` box heights per step so one wild frame cannot
        send the track flying across the image.
        """
        if self.mean is None:
            return
        elapsed = max(float(self.dt_since_update), 1.0)
        dx = (float(new_xyah[0]) - float(self.mean[0])) / elapsed
        dy = (float(new_xyah[1]) - float(self.mean[1])) / elapsed
        dh = (float(new_xyah[3]) - float(self.mean[3])) / elapsed

        limit = max_ratio * max(float(self.mean[3]), 1.0)
        mag = float(np.hypot(dx, dy))
        if mag > limit and mag > 1e-9:
            k = limit / mag
            dx, dy = dx * k, dy * k

        self.mean[4] += dx
        self.mean[5] += dy
        self.mean[7] += dh

    def re_activate(self, new_track, frame_id, new_id=False,
                    seed_velocity=False, reseed_after_gap=3.0, seed_max_ratio=1.5):
        new_xyah = self.tlwh_to_xyah(new_track.tlwh)

        if seed_velocity and (self.hits == 0 or self.dt_since_update >= reseed_after_gap):
            self._seed_velocity(new_xyah, seed_max_ratio)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_xyah
        )
        self.flag_fdf = new_track.flag_fdf
        self.detbb = np.asarray(new_track.detbb, dtype=float) if new_track.detbb is not None else None
        self.landmarks = np.asarray(new_track.landmarks, dtype=float) if new_track.landmarks is not None else None

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.hits += 1
        self.miss_count = 0
        self.dt_since_update = 0.0

    def update(self, new_track, frame_id, seed_velocity=False, seed_max_ratio=1.5):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.flag_fdf = new_track.flag_fdf

        self.landmarks = np.asarray(new_track.landmarks, dtype=float) if new_track.landmarks is not None else None
        self.detbb = np.asarray(new_track.detbb, dtype=float) if new_track.detbb is not None else None

        new_xyah = self.tlwh_to_xyah(new_track.tlwh)

        if seed_velocity and self.hits == 0:
            self._seed_velocity(new_xyah, seed_max_ratio)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_xyah)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        self.hits += 1
        self.miss_count = 0
        self.dt_since_update = 0.0

    # ---------------- geometry ----------------
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def center(self):
        t = self.tlwh
        return np.array([t[0] + t[2] * 0.5, t[1] + t[3] * 0.5], dtype=float)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# ====================================================================
# Config
#
# video_processor.py's existing TrackerConfig keeps working untouched - every
# knob below is read with getattr(args, name, default). Use this class instead
# if you want the new knobs in one obvious place.
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

                 # ---- FIX 1/3: young-track survival ----
                 predict_unconfirmed=True,
                 unconfirmed_thresh=0.9,   # was 0.7 - the gate that killed plates
                 unconfirmed_max_miss=5,   # was 0 - deleted on the first miss

                 # ---- FIX 2: new-track motion ----
                 new_track_vel_std_scale=3.0,
                 seed_velocity_on_first_update=True,
                 seed_max_ratio=1.5,
                 reseed_after_gap=3.0,

                 # ---- FIX 5: recovery pass ----
                 recovery_enabled=True,
                 recovery_thresh=0.7,
                 recovery_expansion=0.5,
                 recovery_base_radius=1.5,
                 recovery_radius_growth=0.4,
                 recovery_max_radius=4.0,
                 recovery_shape_weight=0.3,
                 recovery_class_penalty=0.15,

                 # ---- FIX 6: camera-jolt compensation ----
                 gmc_enabled=True,
                 gmc_min_pairs=2,
                 gmc_max_shift_ratio=0.25,  # of frame height

                 # ---- FIX 7 ----
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

        self.gmc_enabled = gmc_enabled
        self.gmc_min_pairs = gmc_min_pairs
        self.gmc_max_shift_ratio = gmc_max_shift_ratio

        self.max_removed_history = max_removed_history


# Backwards-compatible alias: the old TrackerConfig signature, same defaults
# as video_processor.py already uses.
TrackerConfig = PlateTrackerConfig


# ====================================================================
# BYTETracker
# ====================================================================
class BYTETracker(object):
    def __init__(self, args, frame_rate=30, name=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = deque(maxlen=512)  # FIX 7: bounded

        self.frame_id = 0
        self.args = args
        self.name = str(name) if name is not None else "?"
        self.log = logging.getLogger("plate_tracker.%s" % self.name)

        g = lambda k, d: getattr(args, k, d)  # noqa: E731

        # stock knobs
        self.track_thresh = float(g("track_thresh", 0.5))
        self.match_thresh = float(g("match_thresh", 0.99))
        self.track_buffer = int(g("track_buffer", 60))
        self.mot20 = bool(g("mot20", False))
        self.second_thresh = float(g("second_thresh", 0.5))
        self.duplicate_thresh = float(g("duplicate_thresh", 0.15))

        # FIX 1 / 3
        self.predict_unconfirmed = bool(g("predict_unconfirmed", True))
        self.unconfirmed_thresh = float(g("unconfirmed_thresh", 0.9))
        self.unconfirmed_max_miss = int(g("unconfirmed_max_miss", 5))

        # FIX 2
        self.new_track_vel_std_scale = float(g("new_track_vel_std_scale", 3.0))
        self.seed_velocity = bool(g("seed_velocity_on_first_update", True))
        self.seed_max_ratio = float(g("seed_max_ratio", 1.5))
        self.reseed_after_gap = float(g("reseed_after_gap", 3.0))

        # FIX 5
        self.recovery_enabled = bool(g("recovery_enabled", True))
        self.recovery_thresh = float(g("recovery_thresh", 0.7))
        self.recovery_expansion = float(g("recovery_expansion", 0.5))
        self.recovery_base_radius = float(g("recovery_base_radius", 1.5))
        self.recovery_radius_growth = float(g("recovery_radius_growth", 0.4))
        self.recovery_max_radius = float(g("recovery_max_radius", 4.0))
        self.recovery_shape_weight = float(g("recovery_shape_weight", 0.3))
        self.recovery_class_penalty = float(g("recovery_class_penalty", 0.15))

        # FIX 6
        self.gmc_enabled = bool(g("gmc_enabled", True))
        self.gmc_min_pairs = int(g("gmc_min_pairs", 2))
        self.gmc_max_shift_ratio = float(g("gmc_max_shift_ratio", 0.25))

        # FIX 7
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
            "refind_recovery": 0,
            "removed_unconfirmed": 0,
            "removed_timeout": 0,
            "gmc_applied": 0,
        }

    # ---------------- public helpers ----------------
    def get_stats(self):
        """Counters for a periodic summary line in video_processor."""
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

    # ---------------- main entry point ----------------
    def update(self, output_results, img_info, img_size, dt=None):
        """
        output_results : Nx6 array [x1, y1, x2, y2, score, class_id]
        img_info       : (h, w) of the frame the boxes came from
        img_size       : (h, w) the boxes should be scaled to
        dt             : how many camera frames elapsed since the previous
                         update. Pass the real value (1 + skipped frames) and
                         the motion model stops being wrong whenever the engine
                         loop runs slower than the stream. Defaults to 1.
        """
        self.frame_id += 1
        step_dt = 1.0 if dt is None else max(float(dt), 1e-3)

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ---------------- parse detections ----------------
        # (copy: `bboxes /= scale` below must not mutate the caller's array)
        if output_results is None or len(output_results) == 0:
            output_results = np.zeros((0, 6), dtype=np.float64)
        output_results = np.asarray(output_results, dtype=np.float64).copy()
        if output_results.ndim == 1:
            output_results = (output_results.reshape(1, -1)
                              if output_results.size >= 5
                              else np.zeros((0, 6), dtype=np.float64))

        # columns: [x1, y1, x2, y2, score, class_id, (landmarks...)]
        # exactly what video_processor.py builds with
        # np.column_stack([boxes, confs, clss])
        if output_results.shape[1] >= 6:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            flags = output_results[:, 5]
            landmarks = output_results[:, 6:] if output_results.shape[1] > 6 else None
        elif output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            flags = np.zeros_like(scores)
            landmarks = None
        else:
            raise ValueError(f"Unexpected detection shape: {output_results.shape}")

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        flags_keep = flags[remain_inds]

        if landmarks is not None:
            landmarks_keep = landmarks[remain_inds]
        else:
            landmarks_keep = [None] * len(dets)

        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), score, flag, detbb=tlbr, landmarks=lm)
                for tlbr, score, flag, lm in zip(dets, scores_keep, flags_keep, landmarks_keep)
            ]
        else:
            detections = []

        # ---------------- split confirmed / unconfirmed ----------------
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ================================================================
        # Step 1: predict
        #
        # FIX 1: unconfirmed tracks are predicted too. Stock ByteTrack leaves
        # them frozen at their birth position, which is why one-frame-old
        # tracks could never survive a fast car at a low engine fps.
        # ================================================================
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        if self.predict_unconfirmed and unconfirmed:
            STrack.multi_predict(strack_pool + unconfirmed, step_dt)
        else:
            STrack.multi_predict(strack_pool, step_dt)

        # snapshot predicted geometry before any track gets updated
        pool_boxes = np.array([t.tlbr for t in strack_pool], dtype=float).reshape(-1, 4)
        unc_boxes = np.array([t.tlbr for t in unconfirmed], dtype=float).reshape(-1, 4)
        det_boxes = np.array([d.tlbr for d in detections], dtype=float).reshape(-1, 4)
        det_scores = np.array([d.score for d in detections], dtype=float)
        det_classes = np.array([int(round(float(d.flag_fdf))) for d in detections], dtype=int)

        gmc_residuals = []   # (det_centre - predicted_track_centre) for matched pairs

        def _centre(box):
            return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=float)

        # ================================================================
        # Step 2: first association, high-score detections, IoU + fuse_score
        # ================================================================
        dists = iou_distance_boxes(pool_boxes, det_boxes)
        if not self.mot20:
            dists = fuse_score_array(dists, det_scores)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        matched_pool = set()
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            gmc_residuals.append(_centre(det_boxes[idet]) - _centre(pool_boxes[itracked]))
            matched_pool.add(int(itracked))
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,
                             seed_velocity=self.seed_velocity,
                             seed_max_ratio=self.seed_max_ratio)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  seed_velocity=self.seed_velocity,
                                  reseed_after_gap=self.reseed_after_gap,
                                  seed_max_ratio=self.seed_max_ratio)
                refind_stracks.append(track)
                self.stats["refind_iou"] += 1
                self.log.debug(
                    "[TRK-REFIND] id=%s via=iou missed=%.0f steps conf=%.2f",
                    track.track_id, track.dt_since_update, float(det.score))

        # ================================================================
        # Step 3: second association, low-score detections, plain IoU
        # ================================================================
        if len(dets_second) > 0:
            detections_second = []
            for det in output_results[inds_second]:
                tlwh = STrack.tlbr_to_tlwh(det[:4])
                score = det[4]
                flag = det[5] if det.shape[0] > 5 else 0
                lm = det[6:] if det.shape[0] > 6 else None
                detbb = det[:4]
                detections_second.append(STrack(tlwh, score, flag, detbb=detbb, landmarks=lm))
        else:
            detections_second = []

        r_map = [int(i) for i in u_track if strack_pool[int(i)].state == TrackState.Tracked]
        r_tracked_stracks = [strack_pool[i] for i in r_map]
        r_boxes = pool_boxes[r_map] if len(r_map) else np.zeros((0, 4))
        second_boxes = np.array([d.tlbr for d in detections_second], dtype=float).reshape(-1, 4)

        dists = iou_distance_boxes(r_boxes, second_boxes)
        matches, u_track_second, u_detection_second = linear_assignment(
            dists, thresh=self.second_thresh)
        for ir, idet in matches:
            pool_idx = r_map[int(ir)]
            track = strack_pool[pool_idx]
            det = detections_second[idet]
            gmc_residuals.append(_centre(second_boxes[idet]) - _centre(pool_boxes[pool_idx]))
            matched_pool.add(pool_idx)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,
                             seed_velocity=self.seed_velocity,
                             seed_max_ratio=self.seed_max_ratio)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  seed_velocity=self.seed_velocity,
                                  reseed_after_gap=self.reseed_after_gap,
                                  seed_max_ratio=self.seed_max_ratio)
                refind_stracks.append(track)
                self.stats["refind_iou"] += 1

        u_det_high = [int(i) for i in u_detection]
        still_unmatched_pool = [int(i) for i in u_track if int(i) not in matched_pool]

        # ================================================================
        # Step 3.5: camera-jolt estimate (FIX 6)
        #
        # Two sources of evidence, in order of reliability:
        #   1. the tracks that DID match - their prediction errors all point
        #      the same way when the camera moved, so take the median.
        #   2. if nothing matched (which is exactly what a big jolt causes),
        #      let the unmatched tracks and detections vote on the offset that
        #      would align them. Two objects agreeing is a camera shift; one
        #      object alone is just a car driving, so that returns nothing.
        # ================================================================
        shift = None
        if self.gmc_enabled:
            max_shift = self.gmc_max_shift_ratio * float(img_h)
            shift = estimate_global_shift(gmc_residuals,
                                          min_pairs=self.gmc_min_pairs,
                                          max_shift=max_shift)
            source = "matched"
            if shift is None and u_det_high:
                vote_boxes = pool_boxes[still_unmatched_pool] if still_unmatched_pool \
                    else np.zeros((0, 4))
                if len(unc_boxes):
                    vote_boxes = np.vstack([vote_boxes, unc_boxes])
                shift = vote_global_shift(vote_boxes, det_boxes[u_det_high],
                                          min_support=self.gmc_min_pairs,
                                          max_shift=max_shift)
                source = "vote"
            if shift is not None and (abs(shift[0]) > 1.0 or abs(shift[1]) > 1.0):
                self.stats["gmc_applied"] += 1
                self.log.debug("[GMC] shift=(%.1f, %.1f) via=%s",
                               shift[0], shift[1], source)
            else:
                shift = None

        # ================================================================
        # Step 3.6: RECOVERY pass (FIX 5)
        #
        # This is the one that stops plates being forgotten. Everything above
        # needs the predicted box and the detection to physically overlap.
        # Here we ask a different question: is there an unmatched detection
        # that is near enough, the right size, and the right class to be the
        # car we just lost? No overlap required.
        # ================================================================
        if self.recovery_enabled and still_unmatched_pool and u_det_high:
            rec_boxes = pool_boxes[still_unmatched_pool].copy()
            if shift is not None:
                rec_boxes[:, [0, 2]] += shift[0]
                rec_boxes[:, [1, 3]] += shift[1]

            rec_tracks = [strack_pool[i] for i in still_unmatched_pool]
            rec_missing = [max(t.dt_since_update, 1.0) for t in rec_tracks]
            rec_classes = [int(round(float(t.flag_fdf))) for t in rec_tracks]

            cand_boxes = det_boxes[u_det_high]
            cand_classes = det_classes[u_det_high]

            rcost = recovery_distance(
                rec_boxes, cand_boxes, rec_missing,
                track_classes=rec_classes, det_classes=cand_classes,
                expansion=self.recovery_expansion,
                base_radius=self.recovery_base_radius,
                radius_growth=self.recovery_radius_growth,
                max_radius=self.recovery_max_radius,
                shape_weight=self.recovery_shape_weight,
                class_penalty=self.recovery_class_penalty)

            rmatches, r_u_track, r_u_det = linear_assignment(rcost, thresh=self.recovery_thresh)

            for ir, idc in rmatches:
                pool_idx = still_unmatched_pool[int(ir)]
                det_idx = u_det_high[int(idc)]
                track = strack_pool[pool_idx]
                det = detections[det_idx]
                cost = float(rcost[int(ir), int(idc)])
                iou_now = float(1.0 - iou_distance_boxes(
                    pool_boxes[pool_idx:pool_idx + 1], det_boxes[det_idx:det_idx + 1])[0, 0])

                matched_pool.add(pool_idx)
                track.recovered += 1
                was_lost = (track.state != TrackState.Tracked)
                if was_lost:
                    track.re_activate(det, self.frame_id, new_id=False,
                                      seed_velocity=self.seed_velocity,
                                      reseed_after_gap=self.reseed_after_gap,
                                      seed_max_ratio=self.seed_max_ratio)
                    refind_stracks.append(track)
                else:
                    track.update(det, self.frame_id,
                                 seed_velocity=self.seed_velocity,
                                 seed_max_ratio=self.seed_max_ratio)
                    activated_starcks.append(track)

                self.stats["refind_recovery"] += 1
                self.log.info(
                    "[TRK-REFIND] id=%s via=recovery cost=%.2f iou=%.2f missed=%.0f "
                    "steps conf=%.2f%s",
                    track.track_id, cost, iou_now,
                    max(track.dt_since_update, 0.0), float(det.score),
                    "" if shift is None else " gmc=(%.0f,%.0f)" % shift)

            # detections still unclaimed after the rescue
            u_det_high = [u_det_high[int(i)] for i in r_u_det]
            still_unmatched_pool = [still_unmatched_pool[int(i)] for i in r_u_track]

        # ---- tracks that found nothing anywhere become lost ----
        for pool_idx in still_unmatched_pool:
            track = strack_pool[pool_idx]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                track.miss_count += 1
                lost_stracks.append(track)
                self.stats["lost"] += 1
                self.log.debug(
                    "[TRK-LOST] id=%s age=%df hits=%d - no detection matched",
                    track.track_id, self.frame_id - track.birth_frame, track.hits)

        # ================================================================
        # Step 4: unconfirmed (young) tracks
        #
        # FIX 3: the gate is loosened, the recovery pass applies here too, and
        # a miss no longer kills the track outright - it gets
        # `unconfirmed_max_miss` chances first. This is the change that keeps
        # a plate alive long enough to reach MIN_SEEN_FRAMES downstream.
        # ================================================================
        remaining_dets = [detections[i] for i in u_det_high]
        remaining_boxes = det_boxes[u_det_high] if u_det_high else np.zeros((0, 4))
        remaining_scores = det_scores[u_det_high] if u_det_high else np.zeros((0,))
        remaining_classes = det_classes[u_det_high] if u_det_high else np.zeros((0,), dtype=int)

        dists = iou_distance_boxes(unc_boxes, remaining_boxes)
        if not self.mot20:
            dists = fuse_score_array(dists, remaining_scores)
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=self.unconfirmed_thresh)

        claimed_dets = set()
        for itracked, idet in matches:
            track = unconfirmed[int(itracked)]
            track.update(remaining_dets[int(idet)], self.frame_id,
                         seed_velocity=self.seed_velocity,
                         seed_max_ratio=self.seed_max_ratio)
            activated_starcks.append(track)
            claimed_dets.add(int(idet))
            self.stats["confirmed"] += 1
            self.log.debug("[TRK-CONFIRM] id=%s after=%df hits=%d",
                           track.track_id, self.frame_id - track.birth_frame, track.hits)

        u_unconfirmed = [int(i) for i in u_unconfirmed]
        u_det_left = [int(i) for i in u_detection if int(i) not in claimed_dets]

        # recovery for young tracks as well - this is where the bump case bites
        if self.recovery_enabled and u_unconfirmed and u_det_left:
            rb = unc_boxes[u_unconfirmed].copy()
            if shift is not None:
                rb[:, [0, 2]] += shift[0]
                rb[:, [1, 3]] += shift[1]

            utracks = [unconfirmed[i] for i in u_unconfirmed]
            umissing = [max(t.dt_since_update, 1.0) for t in utracks]
            uclasses = [int(round(float(t.flag_fdf))) for t in utracks]

            rcost = recovery_distance(
                rb, remaining_boxes[u_det_left], umissing,
                track_classes=uclasses, det_classes=remaining_classes[u_det_left],
                expansion=self.recovery_expansion,
                base_radius=self.recovery_base_radius,
                radius_growth=self.recovery_radius_growth,
                max_radius=self.recovery_max_radius,
                shape_weight=self.recovery_shape_weight,
                class_penalty=self.recovery_class_penalty)

            rmatches, r_u_unc, r_u_det = linear_assignment(rcost, thresh=self.recovery_thresh)
            for iu, idc in rmatches:
                track = unconfirmed[u_unconfirmed[int(iu)]]
                det = remaining_dets[u_det_left[int(idc)]]
                track.update(det, self.frame_id,
                             seed_velocity=self.seed_velocity,
                             seed_max_ratio=self.seed_max_ratio)
                track.recovered += 1
                activated_starcks.append(track)
                self.stats["refind_recovery"] += 1
                self.stats["confirmed"] += 1
                self.log.info(
                    "[TRK-REFIND] id=%s via=recovery-young cost=%.2f age=%df conf=%.2f",
                    track.track_id, float(rcost[int(iu), int(idc)]),
                    self.frame_id - track.birth_frame, float(det.score))

            u_unconfirmed = [u_unconfirmed[int(i)] for i in r_u_unc]
            u_det_left = [u_det_left[int(i)] for i in r_u_det]

        # ---- young tracks that matched nothing: grace period, not death ----
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.miss_count += 1
            if track.miss_count > self.unconfirmed_max_miss:
                track.mark_removed()
                removed_stracks.append(track)
                self.stats["removed_unconfirmed"] += 1
                # A plate that never got a second chance. If you see a lot of
                # these, raise unconfirmed_max_miss / recovery_max_radius.
                self.log.info(
                    "[TRK-REMOVE] id=%s reason=unconfirmed_expired age=%df hits=%d misses=%d",
                    track.track_id, self.frame_id - track.birth_frame,
                    track.hits, track.miss_count)
            else:
                self.log.debug(
                    "[TRK-COAST] id=%s unconfirmed miss=%d/%d age=%df",
                    track.track_id, track.miss_count, self.unconfirmed_max_miss,
                    self.frame_id - track.birth_frame)

        # ================================================================
        # Step 5: init brand new tracks from what is left
        # ================================================================
        for idx in u_det_left:
            track = remaining_dets[idx]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id,
                           vel_std_scale=self.new_track_vel_std_scale)
            activated_starcks.append(track)
            self.stats["created"] += 1

            if not self.log.isEnabledFor(logging.DEBUG):
                continue

            # diagnostic: was there a lost track nearby that we *just* failed to
            # rescue? if so the gate is too tight - the numbers tell you by how
            # much.
            near_id, near_d = None, None
            if self.lost_stracks:
                lb = np.array([t.tlbr for t in self.lost_stracks], dtype=float).reshape(-1, 4)
                c_new = _centre(track.tlbr)
                c_old = np.stack([(lb[:, 0] + lb[:, 2]) * 0.5, (lb[:, 1] + lb[:, 3]) * 0.5], axis=1)
                hgt = np.maximum(lb[:, 3] - lb[:, 1], 1.0)
                nd = np.linalg.norm(c_old - c_new[None, :], axis=1) / hgt
                j = int(np.argmin(nd))
                near_id, near_d = self.lost_stracks[j].track_id, float(nd[j])

            self.log.debug(
                "[TRK-NEW] id=%s fid=%d cls=%d conf=%.2f box=(%d,%d,%d,%d)%s",
                track.track_id, self.frame_id, int(round(float(track.flag_fdf))),
                float(track.score), *[int(v) for v in track.tlbr],
                "" if near_id is None else
                " nearest_lost=%s ndist=%.2f" % (near_id, near_d))

        # ================================================================
        # Step 6: retire lost tracks that ran out of buffer
        # ================================================================
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                self.stats["removed_timeout"] += 1
                self.log.debug(
                    "[TRK-REMOVE] id=%s reason=lost_timeout hits=%d recovered=%d",
                    track.track_id, track.hits, track.recovered)

        # ================================================================
        # Step 7: merge the bookkeeping lists
        # ================================================================
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, list(self.removed_stracks))
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, self.duplicate_thresh)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


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
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < thresh)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
