"""
liveness.py (detector)
--------------------------------------------------------------------
Presentation-attack detection: is this a live human face, or a face
being shown to the camera on a printed photo / phone / monitor?

No deep learning, no extra model, no extra weights to ship. Everything
here runs on data the engine already has (the ROI frame, the head
bbox, the 14 YOLO landmarks, and the solvePnP pose) plus a sparse
optical-flow track that costs well under a millisecond per face.

Three independent checks, combined into one verdict per track.

1. PLANARITY (primary, authoritative)
   A flat object — paper, phone, monitor — maps from one frame to any
   other by a SINGLE homography, no matter how it is tilted or moved.
   A real head does not:
     * parallax: as the head turns, nose and ears shift relative to
       the eye plane, which no homography can explain;
     * non-rigid motion: talking and expression move mouth and cheeks.
   So: track sparse features inside the face box with LK, fit a
   RANSAC homography from the anchor frame to now, and measure the
   median reprojection error normalized by face width. Low error +
   high inlier ratio => planar => spoof. This is the only check that
   catches BOTH printed photos and screen replays, because a video of
   a real person playing on a phone is still a plane.

2. CARRIED SURFACE (supporting)
   Extend the same flow to a ring around the face (expanded box minus
   face box) and ask whether those points obey the FACE's homography.
   * photo / phone: the paper edge, the bezel and the holding fingers
     are part of the same rigid plane, so they follow it;
   * real person: that ring is background (static, or moving
     differently) plus hair and neck, which do not follow the face
     plane under rotation.
   Catches bezels, cropped prints and hands without looking for
   straight lines, so bezel-less phones do not defeat it.

3. LANDMARK RIGIDITY (supporting, and the reason this file does not
   simply give up when the head is held still)
   Procrustes-align the landmarks at t against t-k (Umeyama
   similarity: remove translation, scale and rotation) and measure the
   leftover per-point residual. A live face is non-rigid — blinks,
   mouth, brow — so the residual stays clearly above tracker noise. A
   PRINTED PHOTO is perfectly rigid and collapses to ~0.
   Note this check cannot stand alone: a screen replaying a talking
   person is non-rigid too. That is exactly why planarity stays
   authoritative and this one only (a) separates `printed_photo` from
   `screen_replay` in the reason string, and (b) supplies evidence in
   the case check 1 is gated out because the subject never turned.

GATING — judge only when there was something to see. Check 1 needs
real evidence: either the pose changed by >= LIVENESS_MIN_POSE_DELTA_DEG
across the window, or the landmarks visibly deformed. Bbox translation
is deliberately NOT accepted as evidence: sliding a photo across the
frame proves nothing. With no evidence, the track reports
`insufficient_evidence` rather than guessing.

The verdict is sticky and accumulative per track: every evaluation
casts a weighted vote, and the running tally is what gets attached to
the track. A track therefore hardens from `insufficient_evidence` to a
real verdict as the subject moves, and a single noisy frame cannot
flip an established call.

Fail-safe by construction: every public entry point swallows its own
exceptions. If anything in here breaks, the track is reported as
`unknown` and the detection pipeline continues untouched.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("detector.liveness")

# Verdict labels (also what lands in the backend payload)
REAL = "real"
FAKE = "fake"
INSUFFICIENT = "insufficient_evidence"
UNKNOWN = "unknown"
DISABLED = "disabled"


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class LivenessConfig:
    """Env snapshot. Plain attributes so it pickles across the spawn
    boundary into the engine subprocess without surprises."""

    def __init__(self):
        self.enabled = _bool("LIVENESS_ENABLED", "true")

        # Run the flow/homography work every Nth processed frame. The
        # check does not need every frame — it needs a big enough
        # baseline, which is what the anchor gives it.
        self.eval_every_n = max(1, _int("LIVENESS_EVAL_EVERY_N_FRAMES", 3))

        # Compare frame t against an anchor frame t-k rather than the
        # previous frame: adjacent frames move too little for the
        # residual to mean anything.
        self.anchor_min_age = _int("LIVENESS_ANCHOR_MIN_AGE_FRAMES", 5)
        self.anchor_max_age = _int("LIVENESS_ANCHOR_MAX_AGE_FRAMES", 45)

        # Evidence gate: how much pose change counts as "the head
        # actually turned".
        self.min_pose_delta_deg = _float("LIVENESS_MIN_POSE_DELTA_DEG", 10.0)

        # Feature budget inside the face box.
        self.max_points = _int("LIVENESS_MAX_POINTS", 80)
        self.min_points = _int("LIVENESS_MIN_POINTS", 12)

        # Planarity: median reprojection error as a FRACTION of face
        # width. Below this => the motion is explained by one
        # homography => flat. Calibrate on your own footage; 0.006 is
        # a deliberately conservative starting point (a photo sits
        # around 0.003-0.005, which is tracker noise).
        self.planar_residual_thr = _float("LIVENESS_PLANAR_RESIDUAL_THR", 0.006)
        self.planar_inlier_thr = _float("LIVENESS_PLANAR_INLIER_THR", 0.80)

        # Carried surface: fraction of ring points that follow the
        # face's homography. Above this => the surroundings travel
        # with the face => being carried.
        self.ring_scale = _float("LIVENESS_RING_SCALE", 1.6)
        self.ring_follow_thr = _float("LIVENESS_RING_FOLLOW_THR", 0.60)
        self.ring_min_points = _int("LIVENESS_RING_MIN_POINTS", 8)

        # Rigidity: Procrustes residual / face width. Below this the
        # face did not deform at all => print-like.
        self.rigid_residual_thr = _float("LIVENESS_RIGID_RESIDUAL_THR", 0.004)
        self.landmark_conf_thr = _float("LIVENESS_LANDMARK_CONF_THR", 0.3)

        # How many completed evaluations before we are willing to
        # publish anything other than insufficient_evidence.
        self.min_evals = _int("LIVENESS_MIN_EVALS", 3)

        # How many PLANARITY evaluations (check 1, the authoritative
        # one) must have happened before a `real` verdict is allowed.
        #
        # This is a safety interlock, not a tuning knob. The supporting
        # checks are deliberately asymmetric — "ring does not follow"
        # and "face deforms" are only WEAK evidence of life (a screen
        # replaying a talking person deforms too, and a photo held
        # against a busy background has a non-following ring), whereas
        # their negatives are STRONG evidence of a spoof. Without this
        # interlock, a spoof held still enough to gate out check 1 can
        # accumulate enough weak positives to be called `real`, which
        # is the one failure direction that actually matters. With it,
        # no planarity evidence => at most `insufficient_evidence`.
        self.min_planar_evals = _int("LIVENESS_MIN_PLANAR_EVALS", 1)

        # Decision margin on the accumulated score in [-1, 1]:
        # >= +margin => real, <= -margin => fake, else undecided.
        self.decision_margin = _float("LIVENESS_DECISION_MARGIN", 0.25)

        # Per-check weights in the running tally. Asymmetric on
        # purpose — see min_planar_evals above: a check that says
        # "spoof" is trusted far more than the same check saying
        # "live", because only planarity can actually prove life.
        self.w_planarity = _float("LIVENESS_W_PLANARITY", 1.0)
        self.w_ring = _float("LIVENESS_W_RING", 0.6)
        self.w_rigidity = _float("LIVENESS_W_RIGIDITY", 0.4)
        # multiplier applied to a SUPPORTING check's positive (live) vote
        self.support_positive_scale = _float("LIVENESS_SUPPORT_POSITIVE_SCALE", 0.35)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------
def _umeyama_residual(src: np.ndarray, dst: np.ndarray, face_w: float) -> Optional[float]:
    """Best similarity transform (translation + rotation + uniform
    scale) from src to dst, then the median leftover per-point
    distance normalized by face width.

    This is the 'did the face itself deform' measure: everything a
    rigid object can do to its own image under small motion is
    absorbed by the similarity fit, so what remains is non-rigidity.
    """
    if src is None or dst is None or len(src) < 4 or len(src) != len(dst):
        return None
    try:
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)

        mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
        s_c, d_c = src - mu_s, dst - mu_d

        var_s = float((s_c ** 2).sum() / len(src))
        if var_s < 1e-9:
            return None

        cov = (d_c.T @ s_c) / len(src)
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[1, 1] = -1.0

        R = U @ S @ Vt
        scale = float((D * np.diag(S)).sum() / var_s)
        t = mu_d - scale * (R @ mu_s)

        proj = (scale * (R @ src.T)).T + t
        err = np.linalg.norm(proj - dst, axis=1)
        return float(np.median(err) / max(1.0, face_w))
    except Exception:
        return None


def _planarity(p_old: np.ndarray, p_new: np.ndarray, face_w: float,
               cfg: LivenessConfig) -> Optional[Tuple[float, float, np.ndarray]]:
    """Median reprojection error (as a fraction of face width) and
    inlier ratio for the best RANSAC homography old->new, plus the
    homography itself so the ring check can reuse it."""
    if p_old is None or p_new is None:
        return None
    if len(p_old) < cfg.min_points or len(p_old) != len(p_new):
        return None
    try:
        H, inl = cv2.findHomography(
            p_old.reshape(-1, 1, 2).astype(np.float32),
            p_new.reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC,
            max(1.0, 0.01 * face_w),
        )
        if H is None:
            return None
        proj = cv2.perspectiveTransform(
            p_old.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        err = np.linalg.norm(proj - p_new.reshape(-1, 2), axis=1) / max(1.0, face_w)
        inlier_ratio = float(inl.mean()) if inl is not None else 0.0
        return float(np.median(err)), inlier_ratio, H
    except Exception:
        return None


def _ring_follows(H: np.ndarray, r_old: np.ndarray, r_new: np.ndarray,
                  face_w: float, cfg: LivenessConfig) -> Optional[float]:
    """Fraction of ring points whose motion is explained by the FACE's
    homography. High => the surroundings are rigidly attached to the
    face plane => the face is being carried on a surface."""
    if H is None or r_old is None or r_new is None:
        return None
    if len(r_old) < cfg.ring_min_points or len(r_old) != len(r_new):
        return None
    try:
        proj = cv2.perspectiveTransform(
            r_old.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)
        err = np.linalg.norm(proj - r_new.reshape(-1, 2), axis=1) / max(1.0, face_w)
        return float(np.mean(err < cfg.planar_residual_thr * 2.0))
    except Exception:
        return None


# --------------------------------------------------------------------
# per-track state
# --------------------------------------------------------------------
class _TrackState:
    __slots__ = ("anchor_gray", "anchor_pts", "anchor_ring", "anchor_fid",
                 "anchor_pose", "anchor_lms", "score", "weight", "evals",
                 "planar_evals", "last_eval_fid", "verdict", "reason",
                 "last_metrics", "pose_seen_min", "pose_seen_max", "created_ts")

    def __init__(self):
        self.anchor_gray: Optional[np.ndarray] = None
        self.anchor_pts: Optional[np.ndarray] = None
        self.anchor_ring: Optional[np.ndarray] = None
        self.anchor_fid: int = -1
        self.anchor_pose: Optional[Dict[str, float]] = None
        self.anchor_lms: Optional[np.ndarray] = None

        self.score: float = 0.0     # signed, + real / - fake
        self.weight: float = 0.0    # total weight cast
        self.evals: int = 0
        self.planar_evals: int = 0   # how many times check 1 actually ran
        self.last_eval_fid: int = -1

        self.verdict: str = INSUFFICIENT
        self.reason: str = "no evidence yet"
        self.last_metrics: Dict[str, Any] = {}

        self.pose_seen_min: Optional[float] = None
        self.pose_seen_max: Optional[float] = None
        self.created_ts: float = time.time()


class LivenessAnalyzer:
    """One per camera, owned by the Engine. Stateful across frames.

    Public surface is two calls:
        update(...)  -> run a step for one track on this frame
        verdict(...) -> current sticky verdict dict for a track
        drop(track_id)
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, camera_id: str, cfg: Optional[LivenessConfig] = None, logger_=None):
        self.camera_id = str(camera_id)
        self.cfg = cfg or LivenessConfig()
        self.logger = logger_ or logger
        self.tracks: Dict[int, _TrackState] = {}
        self.broken = False

    # ---------------- public ----------------
    def drop(self, track_id: int):
        self.tracks.pop(int(track_id), None)

    def verdict(self, track_id: int) -> Dict[str, Any]:
        """Never raises. Always returns a dict safe to attach to meta."""
        if not self.cfg.enabled:
            return {"liveness": DISABLED, "liveness_score": None,
                    "liveness_reason": "disabled by config", "liveness_evals": 0}
        st = self.tracks.get(int(track_id))
        if st is None:
            return {"liveness": INSUFFICIENT, "liveness_score": None,
                    "liveness_reason": "track not analyzed", "liveness_evals": 0}
        norm = (st.score / st.weight) if st.weight > 1e-6 else None
        return {
            "liveness": st.verdict,
            "liveness_score": round(norm, 4) if norm is not None else None,
            "liveness_reason": st.reason,
            "liveness_evals": st.evals,
            "liveness_metrics": st.last_metrics or None,
        }

    def update(self, track_id: int, gray: np.ndarray, bbox: Tuple[int, int, int, int],
               landmarks: Optional[np.ndarray], pose: Optional[Dict[str, float]],
               fid: int) -> Dict[str, Any]:
        """Advance the analysis for one track on this frame.

        gray      : grayscale of the ROI frame (same coords as bbox)
        bbox      : (x1, y1, x2, y2) of the head, in ROI-frame coords
        landmarks : (N,3) [x, y, conf] in ROI-frame coords, or None
        pose      : {"yaw","pitch","roll"} or None
        fid       : current frame id for this camera
        """
        if not self.cfg.enabled or self.broken:
            return self.verdict(track_id)
        try:
            self._update_inner(int(track_id), gray, bbox, landmarks, pose, int(fid))
        except Exception:
            # Never let anti-spoofing take the pipeline down with it.
            self.logger.exception("[liveness][cam=%s] track %s failed; reporting unknown",
                                  self.camera_id, track_id)
            st = self.tracks.get(int(track_id))
            if st is not None:
                st.verdict = UNKNOWN
                st.reason = "analyzer error"
        return self.verdict(track_id)

    # ---------------- internals ----------------
    def _update_inner(self, track_id: int, gray: np.ndarray,
                      bbox: Tuple[int, int, int, int],
                      landmarks: Optional[np.ndarray],
                      pose: Optional[Dict[str, float]], fid: int):
        if gray is None or gray.size == 0:
            return

        st = self.tracks.get(track_id)
        if st is None:
            st = _TrackState()
            self.tracks[track_id] = st

        # remember the pose range this track has ever shown us
        if pose is not None:
            y = float(pose.get("yaw", 0.0))
            st.pose_seen_min = y if st.pose_seen_min is None else min(st.pose_seen_min, y)
            st.pose_seen_max = y if st.pose_seen_max is None else max(st.pose_seen_max, y)

        if (fid % self.cfg.eval_every_n) != 0:
            return

        x1, y1, x2, y2 = [int(v) for v in bbox]
        H_img, W_img = gray.shape[:2]
        x1 = max(0, min(x1, W_img - 2)); x2 = max(x1 + 2, min(x2, W_img))
        y1 = max(0, min(y1, H_img - 2)); y2 = max(y1 + 2, min(y2, H_img))
        face_w = float(max(2, x2 - x1))

        # ---- (re)anchor when we have nothing usable ----
        need_anchor = (
            st.anchor_gray is None
            or st.anchor_pts is None
            or len(st.anchor_pts) < self.cfg.min_points
            or (fid - st.anchor_fid) > self.cfg.anchor_max_age
        )
        if need_anchor:
            self._set_anchor(st, gray, (x1, y1, x2, y2), landmarks, pose, fid)
            return

        if (fid - st.anchor_fid) < self.cfg.anchor_min_age:
            return

        # ---- track the anchor's points into this frame ----
        face_new, face_old = self._flow(st.anchor_gray, gray, st.anchor_pts)
        if face_new is None or len(face_new) < self.cfg.min_points:
            self._set_anchor(st, gray, (x1, y1, x2, y2), landmarks, pose, fid)
            return

        ring_new, ring_old = self._flow(st.anchor_gray, gray, st.anchor_ring)

        # ---- evidence gate ----
        pose_delta = self._pose_delta(st.anchor_pose, pose)
        rigid_res = self._rigidity(st.anchor_lms, landmarks, face_w)

        deformed = rigid_res is not None and rigid_res > self.cfg.rigid_residual_thr
        turned = pose_delta is not None and pose_delta >= self.cfg.min_pose_delta_deg

        if not turned and not deformed:
            # Nothing happened worth judging. Deliberately do NOT count
            # bbox translation as evidence — sliding a photo across the
            # frame proves nothing about it.
            st.last_metrics = {
                "pose_delta_deg": round(pose_delta, 2) if pose_delta is not None else None,
                "rigid_residual": round(rigid_res, 5) if rigid_res is not None else None,
                "gated": "no pose change, no deformation",
            }
            if st.evals == 0:
                st.verdict = INSUFFICIENT
                st.reason = "subject has not turned or changed expression yet"
            return

        # ---- check 1: planarity ----
        planar = _planarity(face_old, face_new, face_w, self.cfg)
        metrics: Dict[str, Any] = {
            "pose_delta_deg": round(pose_delta, 2) if pose_delta is not None else None,
            "rigid_residual": round(rigid_res, 5) if rigid_res is not None else None,
            "points": int(len(face_new)),
            "baseline_frames": int(fid - st.anchor_fid),
        }

        votes: List[Tuple[float, float, str]] = []  # (weight, signed_vote, note)

        if planar is not None:
            residual, inliers, Hm = planar
            metrics["planar_residual"] = round(residual, 5)
            metrics["planar_inliers"] = round(inliers, 3)

            if turned:
                # Only meaningful when the head actually rotated: that is
                # when a real face MUST break the single-homography model.
                flat = (residual < self.cfg.planar_residual_thr
                        and inliers >= self.cfg.planar_inlier_thr)
                st.planar_evals += 1
                if flat:
                    votes.append((self.cfg.w_planarity, -1.0, "planar under rotation"))
                else:
                    # "Not flat" is the logical complement of the flat
                    # test on BOTH counts: the residual is too big AND
                    # RANSAC could not gather the points onto one plane.
                    # When both hold, no homography explains this motion,
                    # which is the strongest live evidence available —
                    # vote it at full strength rather than scaling it by
                    # how far past the threshold it happened to land, or
                    # a single weak supporting vote can cancel the one
                    # check that actually proves depth.
                    decisive = (residual >= self.cfg.planar_residual_thr
                                and inliers < self.cfg.planar_inlier_thr)
                    votes.append((self.cfg.w_planarity, 1.0 if decisive else 0.5,
                                  "parallax under rotation"))

            # ---- check 2: carried surface ----
            follow = _ring_follows(Hm, ring_old, ring_new, face_w, self.cfg)
            if follow is not None:
                metrics["ring_follow"] = round(follow, 3)
                metrics["ring_points"] = int(len(ring_new)) if ring_new is not None else 0
                if follow >= self.cfg.ring_follow_thr:
                    votes.append((self.cfg.w_ring, -1.0, "surroundings move with the face plane"))
                else:
                    # weak: a print against a busy background looks like this too
                    votes.append((self.cfg.w_ring, self.cfg.support_positive_scale,
                                  "surroundings independent of the face"))

        # ---- check 3: rigidity ----
        if rigid_res is not None:
            if rigid_res < self.cfg.rigid_residual_thr:
                # Perfectly rigid: print-like. A screen replay of a
                # talking person would NOT land here, which is why this
                # vote is weighted below planarity.
                votes.append((self.cfg.w_rigidity, -1.0, "face is perfectly rigid"))
            else:
                # weak: a screen replaying a talking person deforms too
                votes.append((self.cfg.w_rigidity, self.cfg.support_positive_scale,
                              "face deforms non-rigidly"))

        if not votes:
            return

        for w, v, _note in votes:
            st.score += w * v
            st.weight += w
        st.evals += 1
        st.last_eval_fid = fid
        st.last_metrics = metrics

        self._decide(st, metrics)

        # roll the anchor forward so the next window is fresh
        self._set_anchor(st, gray, (x1, y1, x2, y2), landmarks, pose, fid)

    def _decide(self, st: _TrackState, metrics: Dict[str, Any]):
        if st.evals < self.cfg.min_evals or st.weight <= 1e-6:
            st.verdict = INSUFFICIENT
            st.reason = f"only {st.evals}/{self.cfg.min_evals} evaluations so far"
            return

        norm = st.score / st.weight

        # SAFETY INTERLOCK: never call something `real` on supporting
        # evidence alone. Only the planarity check can actually prove a
        # face is not a flat surface; without it the honest answer is
        # "I have not seen enough", never "live".
        if norm >= self.cfg.decision_margin and st.planar_evals < self.cfg.min_planar_evals:
            st.verdict = INSUFFICIENT
            st.reason = ("supporting checks lean live, but the subject never turned "
                         "enough to run the planarity test")
            return

        if norm <= -self.cfg.decision_margin:
            st.verdict = FAKE
            # separate the two spoof families for the operator
            rigid = metrics.get("rigid_residual")
            if rigid is not None and rigid < self.cfg.rigid_residual_thr:
                st.reason = "flat and perfectly rigid — printed photo"
            elif metrics.get("ring_follow", 0.0) and metrics["ring_follow"] >= self.cfg.ring_follow_thr:
                st.reason = "flat surface carried with its surroundings — screen or held print"
            else:
                st.reason = "motion fully explained by a single plane — screen replay"
        elif norm >= self.cfg.decision_margin:
            st.verdict = REAL
            st.reason = "parallax and/or non-rigid deformation consistent with a live face"
        else:
            st.verdict = INSUFFICIENT
            st.reason = f"evidence inconclusive (score {norm:+.2f})"

    # ---- low level ----
    def _flow(self, gray_old: np.ndarray, gray_new: np.ndarray,
              pts_old: Optional[np.ndarray]):
        """LK forward+backward; keeps only points that survive both ways."""
        if pts_old is None or len(pts_old) == 0:
            return None, None
        p0 = pts_old.reshape(-1, 1, 2).astype(np.float32)
        p1, stt, _err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None, **self.LK_PARAMS)
        if p1 is None:
            return None, None
        p0r, st2, _e2 = cv2.calcOpticalFlowPyrLK(gray_new, gray_old, p1, None, **self.LK_PARAMS)
        if p0r is None:
            return None, None

        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good = (stt.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.5)
        if good.sum() < 4:
            return None, None
        return p1.reshape(-1, 2)[good], p0.reshape(-1, 2)[good]

    def _set_anchor(self, st: _TrackState, gray: np.ndarray,
                    bbox: Tuple[int, int, int, int],
                    landmarks: Optional[np.ndarray],
                    pose: Optional[Dict[str, float]], fid: int):
        x1, y1, x2, y2 = bbox
        face_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        face_mask[y1:y2, x1:x2] = 255

        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_points, qualityLevel=0.01,
            minDistance=4, mask=face_mask, blockSize=5,
        )
        st.anchor_pts = pts.reshape(-1, 2) if pts is not None else None

        # ring = expanded box minus the face box
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        hw, hh = (x2 - x1) * 0.5 * self.cfg.ring_scale, (y2 - y1) * 0.5 * self.cfg.ring_scale
        H_img, W_img = gray.shape[:2]
        ex1, ey1 = int(max(0, cx - hw)), int(max(0, cy - hh))
        ex2, ey2 = int(min(W_img, cx + hw)), int(min(H_img, cy + hh))

        ring_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        ring_mask[ey1:ey2, ex1:ex2] = 255
        ring_mask[y1:y2, x1:x2] = 0
        rpts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_points, qualityLevel=0.01,
            minDistance=4, mask=ring_mask, blockSize=5,
        )
        st.anchor_ring = rpts.reshape(-1, 2) if rpts is not None else None

        st.anchor_gray = gray.copy()
        st.anchor_fid = fid
        st.anchor_pose = dict(pose) if pose else None
        st.anchor_lms = self._conf_landmarks(landmarks)

    def _conf_landmarks(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if landmarks is None or len(landmarks) == 0:
            return None
        try:
            lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 3)
            keep = lm[:, 2] >= self.cfg.landmark_conf_thr
            if keep.sum() < 4:
                return None
            # keep the mask alongside so t and t-k stay index-aligned
            out = np.full((len(lm), 2), np.nan, dtype=np.float64)
            out[keep] = lm[keep, :2]
            return out
        except Exception:
            return None

    def _rigidity(self, lms_old: Optional[np.ndarray],
                  lms_new_raw: Optional[np.ndarray], face_w: float) -> Optional[float]:
        lms_new = self._conf_landmarks(lms_new_raw)
        if lms_old is None or lms_new is None or len(lms_old) != len(lms_new):
            return None
        both = ~(np.isnan(lms_old).any(axis=1) | np.isnan(lms_new).any(axis=1))
        if both.sum() < 5:
            return None
        return _umeyama_residual(lms_old[both], lms_new[both], face_w)

    @staticmethod
    def _pose_delta(pose_old: Optional[Dict[str, float]],
                    pose_new: Optional[Dict[str, float]]) -> Optional[float]:
        if not pose_old or not pose_new:
            return None
        dy = abs(float(pose_new.get("yaw", 0.0)) - float(pose_old.get("yaw", 0.0)))
        dp = abs(float(pose_new.get("pitch", 0.0)) - float(pose_old.get("pitch", 0.0)))
        return max(dy, dp)
