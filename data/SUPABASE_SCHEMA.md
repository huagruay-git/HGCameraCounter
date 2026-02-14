# Supabase Schema Setup

## Overview

This document describes the tables and RLS policies needed for HG Camera Counter.

---

## Tables

### 1. `device_status` - Device Heartbeat

Tracks real-time status of each device/installation.

```sql
CREATE TABLE device_status (
  id BIGSERIAL PRIMARY KEY,
  device_id TEXT NOT NULL UNIQUE,
  branch_code TEXT NOT NULL,
  status TEXT DEFAULT 'online', -- online, offline, error
  last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  cameras_ok INT DEFAULT 0,
  active_people INT DEFAULT 0,
  haircuts_today INT DEFAULT 0,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_device_status_branch ON device_status(branch_code);
CREATE INDEX idx_device_status_device ON device_status(device_id);
CREATE INDEX idx_device_status_heartbeat ON device_status(last_heartbeat DESC);

-- RLS
ALTER TABLE device_status ENABLE ROW LEVEL SECURITY;

-- Policy: Branch can read own devices
CREATE POLICY "read_own_devices" ON device_status
  FOR SELECT USING (
    auth.jwt() ->> 'branch_code' = branch_code
  );

-- Policy: Device can update own status
CREATE POLICY "update_own_status" ON device_status
  FOR UPDATE USING (
    auth.jwt() ->> 'device_id' = device_id
  );
```

### 2. `events` - Detected Events

Stores all detected events (haircut, wash, wait, entrance, exit).

```sql
CREATE TABLE events (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  camera TEXT NOT NULL,
  event_type TEXT NOT NULL, -- chair, wash, wait, entrance, exit
  person_id INT NOT NULL,
  zone_name TEXT NOT NULL,
  dwell_seconds FLOAT,
  branch_code TEXT NOT NULL,
  device_id TEXT NOT NULL,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_events_branch ON events(branch_code);
CREATE INDEX idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_camera ON events(camera);
CREATE INDEX idx_events_composite ON events(branch_code, timestamp DESC, event_type);

-- RLS
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- Policy: Branch can read own events
CREATE POLICY "read_own_events" ON events
  FOR SELECT USING (
    auth.jwt() ->> 'branch_code' = branch_code
  );

-- Policy: Device can insert events
CREATE POLICY "insert_own_events" ON events
  FOR INSERT WITH CHECK (
    auth.jwt() ->> 'branch_code' = branch_code
  );
```

### 3. `daily_summary` - Daily Reports

Pre-aggregated daily summary (optional, for performance).

```sql
CREATE TABLE daily_summary (
  id BIGSERIAL PRIMARY KEY,
  date DATE NOT NULL,
  branch_code TEXT NOT NULL,
  device_id TEXT NOT NULL,
  
  total_events INT DEFAULT 0,
  haircuts INT DEFAULT 0,
  washes INT DEFAULT 0,
  waits INT DEFAULT 0,
  entrances INT DEFAULT 0,
  exits INT DEFAULT 0,
  
  unique_people INT DEFAULT 0,
  avg_dwell_seconds FLOAT,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  UNIQUE(date, branch_code, device_id)
);

-- Indexes
CREATE INDEX idx_summary_branch_date ON daily_summary(branch_code, date DESC);
CREATE INDEX idx_summary_device_date ON daily_summary(device_id, date DESC);

-- RLS
ALTER TABLE daily_summary ENABLE ROW LEVEL SECURITY;

-- Policy: Branch can read own summaries
CREATE POLICY "read_own_summary" ON daily_summary
  FOR SELECT USING (
    auth.jwt() ->> 'branch_code' = branch_code
  );
```

### 4. `branch_config` - Branch Configuration

Master configuration per branch.

```sql
CREATE TABLE branch_config (
  id BIGSERIAL PRIMARY KEY,
  branch_code TEXT NOT NULL UNIQUE,
  branch_name TEXT,
  
  -- Settings
  dwell_time_chair INT DEFAULT 120,
  dwell_time_wash INT DEFAULT 60,
  dwell_time_wait INT DEFAULT 30,
  
  -- Devices
  device_count INT DEFAULT 1,
  
  -- Features
  enable_reid BOOLEAN DEFAULT FALSE,
  enable_snapshots BOOLEAN DEFAULT TRUE,
  
  metadata JSONB,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RLS
ALTER TABLE branch_config ENABLE ROW LEVEL SECURITY;

-- Policy: Branch can read own config
CREATE POLICY "read_own_config" ON branch_config
  FOR SELECT USING (
    auth.jwt() ->> 'branch_code' = branch_code
  );
```

---

## Authentication

### JWT Claims

JWT token should include:
- `branch_code` - The branch identifier
- `device_id` - Device identifier (optional, for device-specific RLS)
- `role` - 'device', 'manager', 'admin'

### Example JWT Payload

```json
{
  "sub": "device-123",
  "branch_code": "DEMO",
  "device_id": "CAM_01",
  "role": "device",
  "iat": 1234567890,
  "exp": 1234571490
}
```

---

## RLS Policies

### For Devices (Device Service)

Devices can:
- Insert their own events ✓
- Update their own device_status ✓
- Read device_status (own device) ✓
- Read branch_config (own branch) ✓

### For Managers (Controller App)

Managers can:
- Read events (own branch) ✓
- Read device_status (own branch) ✓
- Read daily_summary (own branch) ✓
- Update branch_config (own branch) ✓

### For Admins

Full access to all data.

---

## API Endpoints

### Device Status Update

```
POST /rest/v1/device_status

Body:
{
  "device_id": "CAM_01",
  "branch_code": "DEMO",
  "status": "online",
  "cameras_ok": 2,
  "active_people": 5,
  "haircuts_today": 12,
  "metadata": {}
}
```

### Event Submission

```
POST /rest/v1/events

Body (batch):
[
  {
    "timestamp": "2026-02-12T10:30:00Z",
    "camera": "Camera_01",
    "event_type": "chair",
    "person_id": 42,
    "zone_name": "CHAIR_1",
    "dwell_seconds": 150.5,
    "branch_code": "DEMO",
    "device_id": "CAM_01",
    "metadata": {}
  },
  ...
]
```

### Get Daily Summary

```
GET /rest/v1/daily_summary?branch_code=eq.DEMO&date=eq.2026-02-12

Response:
[
  {
    "date": "2026-02-12",
    "branch_code": "DEMO",
    "device_id": "CAM_01",
    "haircuts": 12,
    "washes": 5,
    "waits": 8,
    "unique_people": 18,
    "avg_dwell_seconds": 125.5
  }
]
```

---

## Monitoring & Maintenance

### Queries for Monitoring

```sql
-- Recent device statuses
SELECT device_id, status, last_heartbeat 
FROM device_status 
WHERE branch_code = 'DEMO'
ORDER BY last_heartbeat DESC;

-- Today's event summary
SELECT 
  event_type,
  COUNT(*) as count,
  AVG(dwell_seconds) as avg_dwell
FROM events
WHERE branch_code = 'DEMO'
  AND DATE(timestamp) = CURRENT_DATE
GROUP BY event_type;

-- Offline devices (no heartbeat in 5 minutes)
SELECT device_id, last_heartbeat
FROM device_status
WHERE branch_code = 'DEMO'
  AND last_heartbeat < NOW() - INTERVAL '5 minutes';
```

### Cleanup Policies

```sql
-- Archive old events (keep 90 days)
DELETE FROM events
WHERE created_at < NOW() - INTERVAL '90 days';

-- Archive old daily summaries (keep 1 year)
DELETE FROM daily_summary
WHERE created_at < NOW() - INTERVAL '365 days';
```

---

## Implementation Steps

1. **Create Tables**
   - Run SQL scripts above in Supabase SQL editor

2. **Enable RLS**
   - All tables have RLS enabled
   - Policies control access by branch_code

3. **Create Indexes**
   - Performance optimization for queries

4. **Configure Auth**
   - Setup JWT token generation
   - Configure device service account

5. **Test**
   - Verify RLS policies work correctly
   - Test event submission
   - Test device status update

---

## Environment Setup

### In `.env` or `data/config/config.yaml`:

```yaml
supabase:
  url: "https://xxxx.supabase.co"
  key: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  branch_code: "DEMO"
```

### Service Account Key

For runtime service (device):
- Ask Supabase admin to create service key with branch_code = DEMO
- Use that key in runtime/agent.py

### For Controller (read-only):
- Use anon key with limited RLS policies
- Only read access to own branch

---

## Troubleshooting

### "RLS policy not met"
- Check JWT token contains correct branch_code
- Verify RLS policies are created

### "Table does not exist"
- Run SQL scripts to create tables
- Check table names match policies

### "Too many events"
- Implement batch submission (implemented in SupabaseSync)
- Archive old events regularly

---

**Version**: 0.1.0  
**Last Updated**: 2026-02-12
