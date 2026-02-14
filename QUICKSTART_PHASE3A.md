# Quick Start - Phase 3A Real-time Dashboard

## Prerequisites

```bash
# Ensure Python 3.8+
python3 --version

# Install dependencies
pip install -r requirements.txt

# Verify YOLO model exists
ls models/yolov8m.pt
```

## Running the System

### Option 1: Full GUI with Real-time Updates

```bash
# 1. Start the controller GUI
python controller/main.py

# 2. In the GUI window:
#    - Navigate to "Setup Wizard" to configure cameras/zones
#    - Once configured, click "▶ Start Service"
#    - Watch "Dashboard" tab for real-time updates

# 3. Dashboard will show:
#    ✓ Camera connection status (every 2s)
#    ✓ Event counts - haircuts/washes/waiting (every 5s)
#    ✓ Active people being tracked (every 2s)
#    ✓ Connection status indicator
```

### Option 2: Test Real-time Updates Only

```bash
# This provides setup prompts and GUI launcher
python test_gui_realtime.py
```

### Option 3: Background Service (No GUI)

```bash
# Start service directly (outputs to logs/)
python runtime/agent_v2.py &

# Monitor logs
tail -f logs/runtime.log
```

---

## What You'll See in Real-time

### Dashboard Tab (Updates Automatically)

```
┌─────────────────────────────────────────────┐
│ HG Camera Counter - Controller              │
├─────────────────────────────────────────────┤
│ Runtime: 🟢 Running                         │
│ Last Heartbeat: 14:32:05                    │
│ Active People: 12                           │
│                                             │
│ Camera Status (Real-time):                  │
│ ├─ Camera_01: ✓ 24.5 fps                   │
│ └─ Camera_02: ✓ 23.8 fps                   │
│                                             │
│ Event Counts (Today):                       │
│ Haircuts: 42                                │
│ Washes: 18                                  │
│ Waiting: 7                                  │
│                                             │
│ Updated: 14:32:15                           │
│ 🟢 Live (auto-updating...)                  │
│ [🔄 Manual Refresh]                         │
└─────────────────────────────────────────────┘
```

### Live Updates

**Every 2 seconds:**
- Camera connection status changes
- FPS updates
- Active people count changes

**Every 5 seconds:**
- Event counts update (haircuts/washes/waiting)
- Updated timestamp shows

**Real-time (as they occur):**
- Individual events logged to service
- Events queued for Supabase

---

## Configuration

### config.yaml (Setup Wizard or Manual)

```yaml
# Basic settings
project_name: "HG Camera Counter"
branch_code: "DEMO"

# Cameras to monitor
cameras:
  Camera_01:
    enabled: true
    rtsp_url: "rtsp://192.168.1.100:554/stream1"
    zones_file: "zones_Camera_01.json"

# YOLO detection
yolo:
  model: "yolov8m.pt"
  device: "auto"  # or "cpu", "mps", "cuda"

# Event detection thresholds
dwell_time:
  haircut: 300  # 5 minutes
  wash: 600     # 10 minutes
  wait: 30      # 30 seconds

# Runtime settings
runtime:
  target_fps: 10
  heartbeat_interval: 30

# Supabase (optional for now)
supabase:
  url: ""
  key: ""
```

### Zones Definition (zones_Camera_01.json)

```json
{
  "CHAIR_A": [
    [0.1, 0.2],
    [0.3, 0.2],
    [0.3, 0.5],
    [0.1, 0.5]
  ],
  "WAIT_AREA": [
    [0.4, 0.1],
    [0.9, 0.1],
    [0.9, 0.9],
    [0.4, 0.9]
  ]
}
```

---

## Troubleshooting

### Dashboard Not Showing Updates

**Check 1: Is service running?**
```bash
# Look for agent_v2.py process
ps aux | grep agent_v2

# Check service logs
tail -50 logs/runtime.log
```

**Check 2: Are cameras connected?**
```bash
# Test RTSP connection
ffprobe "rtsp://192.168.1.100:554/stream1"
# Should show video information, not error
```

**Check 3: Is GUI receiving data?**
```bash
# Enable debug logging in dashboard_client.py
# You'll see "Status update received" in logs
```

### Camera Shows Offline (✗)

**Possible causes:**
1. RTSP URL incorrect in config.yaml
2. Camera not online or unreachable
3. Network firewall blocking port 554
4. Camera requires authentication

**Solution:**
- Edit config.yaml with correct RTSP URL
- Test with ffprobe command above
- Restart service: Stop → Start in GUI

### Event Counts Not Increasing

**Possible causes:**
1. No people detected by YOLO
2. People not staying in zones long enough
3. Zones not defined or overlapping incorrectly

**Debug steps:**
1. Check yolov8m.pt file exists in models/
2. Verify zone definitions in setup wizard
3. Check dwell_time thresholds in config.yaml
4. Monitor logs: `tail -f logs/runtime.log | grep event`

### GUI Frozen or Slow

**Possible causes:**
1. YOLO inference taking too long (GPU issue)
2. Too many messages in queue (network slow)
3. System CPU/memory overloaded

**Solutions:**
1. Reduce target_fps in config (lower from 10 to 5)
2. Reduce number of cameras
3. Close other programs to free resources

---

## What Happens Behind the Scenes

### Timeline of a Single Event

```
14:30:00 - Person enters "CHAIR_A" zone
          └─> EventTracker starts timer

14:30:47 - Person has been in zone 47 seconds
          └─> Exceeds dwell_time: 300s? No, wait more

14:35:05 - Person has been in zone 305 seconds
          └─> Exceeds dwell_time: 300s? YES!
          └─> Event("haircut", Camera_01, CHAIR_A, 305s) generated

14:35:05 - Event added to EventTracker
14:35:06 - submit_events_loop() fetches events
          └─> Broadcasts event to GUI (real-time)
          └─> Adds to Supabase queue (for submission)

14:35:10 - Person leaves zone
          └─> EventTracker closes session

14:35:15 - GUI summary updates
          └─> Haircuts: 42 (was 41)
```

### Data Flow Example

```
event_tracker.get_summary() →
{
  "haircut": 42,
  "wash": 18,
  "wait": 7
}
    ↓
_broadcast_summary(summary)
    ↓
broadcaster.broadcast_summary(DashboardUpdate(...))
    ↓
All subscribers' queues receive message
    ↓
GUIDashboardClient.get_updates() retrieves from queue
    ↓
on_summary_updated.emit(summary) - Qt signal
    ↓
MainController.on_summary_updated() slot executes
    ↓
self.event_counts.setText("Haircuts: 42\nWashes: 18\nWaiting: 7")
    ↓
GUI display updated on screen
```

---

## Performance Tips

### For Better FPS

1. **Reduce YOLO size** (faster but less accurate)
   ```yaml
   yolo:
     model: "yolov8n.pt"  # nano instead of medium
   ```

2. **Lower target FPS** (processes fewer frames)
   ```yaml
   runtime:
     target_fps: 5  # process 5 frames/sec instead of 10
   ```

3. **Use GPU if available**
   ```yaml
   yolo:
     device: "cuda"  # if NVIDIA GPU available
   ```

### For Stability

1. **Add memory guard** (prevent runaway)
   ```bash
   ulimit -v 4000000  # limit to 4GB before starting
   ```

2. **Monitor background processes**
   ```bash
   watch -n 1 'ps aux | grep agent_v2'
   ```

### For Responsiveness

1. **Reduce GUI update intervals** in agent_v2.py
   ```python
   if current_time - self.last_status_broadcast > 1.0:  # 1s instead of 2s
   ```

2. **Increase queue poll frequency** in dashboard_client.py
   ```python
   updates = self.base_client.get_updates(timeout=0.2)  # 0.2s instead of 0.5s
   ```

---

## Logs to Check

```bash
# Runtime service logs
tail -f logs/runtime.log

# Controller GUI logs
tail -f logs/controller.log

# Filter for errors only
grep ERROR logs/runtime.log

# Filter for specific camera
grep "Camera_01" logs/runtime.log

# Show last 100 lines
tail -100 logs/runtime.log
```

---

## Common Commands

```bash
# Kill lingering processes
pkill -f "agent_v2.py"
pkill -f "python controller/main.py"

# Clear logs
rm logs/*.log

# Reset all state (backup first!)
rm -rf logs/ reports/ snapshots/
mkdir -p logs reports snapshots

# Check system resources
top -p $(pgrep -f agent_v2.py)

# Monitor network (if camera streaming)
iftop
```

---

## Next Steps

1. **Configure your cameras** in Setup Wizard
2. **Define zones** for each camera (haircut area, wash area, etc.)
3. **Set dwell times** based on your salon workflow
4. **Run a test** for 1 hour and verify counts make sense
5. **(Phase 3B)** Add RTSP watchdog for reliability
6. **(Phase 3C)** Package for deployment

---

**Version**: Phase 3A
**Last Updated**: February 12, 2026
**Status**: ✅ Ready for Testing
