"""Post-hoc clustering of tracklets into stable identities.

Turns many short, fragmented tracklets into a small set of persistent player IDs.
Two ingredients:

1. Appearance: each tracklet is embedded once (mean DINOv2 feature); we merge by
   cosine distance via agglomerative (average-linkage) clustering.
2. A hard **cannot-link** constraint from the offline structure: if two tracklets
   share even one frame, they are different people and must never be merged. This
   is the kind of global constraint that only an offline pipeline can enforce, and
   it stops co-present people from collapsing into one identity even when their
   appearance features are too weak to separate them.

With a known head count (`num_ids`) we merge until that many clusters remain
(respecting the constraint floor); otherwise we merge until the nearest allowable
pair exceeds `distance_threshold`. scipy only — no sklearn.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _frames(tracklet):
    return {d.frame_idx for d in tracklet.detections}


def _constrained_agglomerative(dist, cannot_link, num_ids=None, threshold=0.25):
    """Average-linkage agglomerative clustering with cannot-link constraints.

    dist: (n,n) distance matrix. cannot_link: (n,n) bool, True => may not co-cluster.
    Returns a list of integer labels (length n). n is small (one per tracklet), so
    the simple O(n^3) recompute is fine.
    """
    n = dist.shape[0]
    members = {i: [i] for i in range(n)}

    def allowed(a, b):
        return not any(cannot_link[i][j] for i in members[a] for j in members[b])

    def linkage(a, b):  # average linkage over original distances
        pairs = [dist[i][j] for i in members[a] for j in members[b]]
        return sum(pairs) / len(pairs)

    next_id = n
    while num_ids is None or len(members) > num_ids:
        best = None
        keys = list(members)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = keys[x], keys[y]
                if not allowed(a, b):
                    continue
                d = linkage(a, b)
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            break  # constraints forbid any further merge
        d, a, b = best
        if num_ids is None and d > threshold:
            break  # nearest allowable pair is too far apart
        members[next_id] = members.pop(a) + members.pop(b)
        next_id += 1

    labels = [0] * n
    for lab, ms in enumerate(members.values()):
        for m in ms:
            labels[m] = lab
    return labels


def cluster_tracklets(tracklets, embedder, num_ids=None, distance_threshold=0.25,
                      min_frames=1):
    """Embed and cluster tracklets. Returns {track_id: player_id} (0-based).

    player_ids are numbered by order of first appearance, so labels are stable and
    readable regardless of internal cluster numbering.
    """
    track_ids, embs, framesets = [], [], []
    for tid, t in tracklets.items():
        if t.num_frames < min_frames:
            continue
        t.embedding = embedder.embed(t.crops)
        track_ids.append(tid)
        embs.append(t.embedding)
        framesets.append(_frames(t))

    if not track_ids:
        return {}
    if len(track_ids) == 1:
        return {track_ids[0]: 0}

    n = len(track_ids)
    dist = squareform(pdist(np.stack(embs), metric="cosine"))
    # Cannot-link: tracklets that share any frame are necessarily different people.
    cannot = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if framesets[i] & framesets[j]:
                cannot[i][j] = cannot[j][i] = True

    labels = _constrained_agglomerative(dist, cannot, num_ids=num_ids,
                                        threshold=distance_threshold)

    floor = _max_overlap(framesets)
    n_found = len(set(labels))
    if num_ids and n_found > num_ids:
        print(f"[cluster] requested {num_ids} ids but {n_found} co-present tracklets "
              f"force at least {floor}; returning {n_found}")

    # Renumber clusters by first appearance for stable, human-readable ids.
    order = sorted(range(n), key=lambda i: tracklets[track_ids[i]].first_frame)
    raw_to_pid, assignment, next_pid = {}, {}, 0
    for i in order:
        raw = labels[i]
        if raw not in raw_to_pid:
            raw_to_pid[raw] = next_pid
            next_pid += 1
        assignment[track_ids[i]] = raw_to_pid[raw]
    return assignment


def _max_overlap(framesets):
    """Largest number of tracklets present in any single frame (a hard floor on
    the number of distinct identities)."""
    counts = {}
    for fs in framesets:
        for f in fs:
            counts[f] = counts.get(f, 0) + 1
    return max(counts.values()) if counts else 0
