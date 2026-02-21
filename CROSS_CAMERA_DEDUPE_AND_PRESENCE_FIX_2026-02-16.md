# Cross-Camera Dedupe and Presence Stability Fix (2026-02-16)

## Problem Summary
- Same person visible in 2 cameras was sometimes counted as 2 haircuts.
- Dashboard `Active People` dropped to `0` too quickly when detector/tracker briefly missed frames.

## Root Causes
- Cross-camera PID canonicalization skipped all candidates when `emb=None` (ReID disabled path), so merge did not happen reliably.
- One camera often had `ids=fallback` (tracker ID unavailable), causing local PID jitter.
- Dashboard used the same short TTL as counting (`customer_active_ttl_sec`), so UI flickered to zero after short miss windows.

## Code Changes Applied

### 1) Cross-camera duplicate count prevention
File: `runtime/agent_v2.py`
- Updated `_canonicalize_pids_cross_camera`:
  - Keep PID candidates even when `emb=None`.
  - Added non-ReID fallback merge rule:
    - If two detections are from different cameras,
    - in near-simultaneous time window,
    - and both are in same business zone family (`CHAIR_*` or `WASH*`),
    - merge to one canonical PID.

### 2) Fallback local ID stability
File: `runtime/agent_v2.py`
- Updated `_assign_fallback_local_vid`:
  - `max_age`: `2.0 -> 3.5`
  - distance threshold: `0.08 -> 0.12`
- Goal: reduce local PID churn when tracker IDs temporarily disappear.

### 3) Continuous presence on dashboard (without changing count logic)
File: `runtime/agent_v2.py`
- Added `runtime.dashboard_presence_ttl_sec` (new config key).
- Counting loop still uses `runtime.customer_active_ttl_sec`.
- Dashboard status filtering and no-detection keepalive now use `dashboard_presence_ttl_sec`.
- Result: UI remains stable during short detection gaps while count logic stays strict.

### 4) Config template updates
File: `data/config/config.template.yaml`
- Added runtime keys:
  - `customer_active_ttl_sec`
  - `dashboard_presence_ttl_sec`
  - `sit_min_sec`
  - `vacant_grace_sec`
  - `zone_point_mode`

## Runtime Config Used (current)
File: `data/config/config.yaml`
- `yolo.mode: track`
- `runtime.sit_min_sec: 2.0`
- `runtime.customer_active_ttl_sec: 10.0`
- `runtime.dashboard_presence_ttl_sec: 25.0`
- `runtime.zone_point_mode: center`

## Verification Evidence
- Runtime now reports successful cross-camera merge:
  - `Canonicalized cross-camera PIDs: merged_groups=1 ...`
- Duplicate haircut from two camera angles reduced to single event:
  - `Queued 1 events for submission`
  - `Summary ... 'haircuts': 1 ...`

## How to Validate Again
1. Start service.
2. Sit one person where both cameras can see at the same time.
3. Confirm log has `Canonicalized cross-camera PIDs`.
4. Confirm only one haircut event is queued/added.
5. Confirm dashboard does not drop to zero immediately during short misses.

## Known Limitation
- If one camera continuously loses tracking (`ids=fallback` always), merge is heuristic (zone/time based) when ReID is off.
- For strongest cross-camera identity, enable ReID with staff/customer embedding pipeline.
