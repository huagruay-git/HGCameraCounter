"""
CSV Report Generator

Generate daily CSV reports from events
"""

import os
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from shared.event_tracker import Event, EventType


class ReportGenerator:
    """Generate CSV reports"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
    
    def save_events_csv(
        self,
        events: List[Event],
        filename: str = None
    ) -> str:
        """
        Save events to CSV file
        
        Args:
            events: List of Event objects
            filename: Output filename (default: daily report)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"report_{today}.csv"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        if not events:
            # Create empty report
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                writer.writeheader()
            return filepath
        
        # Write events
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
            writer.writeheader()
            
            for event in events:
                writer.writerow(event.to_dict())
        
        return filepath
    
    def get_daily_summary(
        self,
        events: List[Event],
        date: datetime = None
    ) -> Dict[str, Any]:
        """Get daily summary statistics"""
        if date is None:
            date = datetime.now()
        
        # Filter events for the day
        start = datetime.combine(date.date(), datetime.min.time())
        end = start + timedelta(days=1)
        
        day_events = [
            e for e in events
            if start <= e.timestamp < end
        ]
        
        # Count by type
        event_counts = {}
        for event_type in EventType:
            count = len([e for e in day_events if e.event_type == event_type])
            event_counts[event_type.value] = count
        
        # Count by camera
        camera_counts = {}
        for event in day_events:
            if event.camera not in camera_counts:
                camera_counts[event.camera] = 0
            camera_counts[event.camera] += 1
        
        # Calculate average dwell times
        avg_dwell = {}
        for event_type in EventType:
            type_events = [e for e in day_events if e.event_type == event_type]
            if type_events:
                avg_dwell[event_type.value] = sum(e.dwell_seconds for e in type_events) / len(type_events)
            else:
                avg_dwell[event_type.value] = 0.0
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_events": len(day_events),
            "event_counts": event_counts,
            "camera_counts": camera_counts,
            "average_dwell_seconds": avg_dwell,
            "unique_people": len(set(e.person_id for e in day_events)),
        }
    
    def generate_daily_report(
        self,
        events: List[Event],
        date: datetime = None
    ) -> str:
        """
        Generate comprehensive daily report
        
        Args:
            events: All events
            date: Report date (default: today)
        
        Returns:
            Path to report file
        """
        if date is None:
            date = datetime.now()
        
        # Filter events for the day
        start = datetime.combine(date.date(), datetime.min.time())
        end = start + timedelta(days=1)
        
        day_events = [
            e for e in events
            if start <= e.timestamp < end
        ]
        
        # Save events CSV
        report_date = date.strftime("%Y-%m-%d")
        csv_file = f"report_{report_date}.csv"
        csv_path = self.save_events_csv(day_events, csv_file)
        
        # Generate summary file
        summary = self.get_daily_summary(events, date)
        
        summary_file = f"summary_{report_date}.txt"
        summary_path = os.path.join(self.reports_dir, summary_file)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Daily Report: {report_date}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Events: {summary['total_events']}\n")
            f.write(f"Unique People: {summary['unique_people']}\n\n")
            
            f.write("Event Breakdown:\n")
            for event_type, count in summary['event_counts'].items():
                f.write(f"  {event_type}: {count}\n")
            
            f.write("\nBy Camera:\n")
            for camera, count in summary['camera_counts'].items():
                f.write(f"  {camera}: {count}\n")
            
            f.write("\nAverage Dwell Time (seconds):\n")
            for event_type, avg_time in summary['average_dwell_seconds'].items():
                f.write(f"  {event_type}: {avg_time:.1f}s\n")
        
        return csv_path
    
    @staticmethod
    def _get_fieldnames() -> List[str]:
        """Get CSV field names"""
        return [
            "timestamp",
            "camera",
            "event_type",
            "person_id",
            "zone_name",
            "dwell_seconds",
            "metadata",
        ]
    
    def get_all_reports(self) -> List[str]:
        """Get list of all report files"""
        if not os.path.exists(self.reports_dir):
            return []
        
        files = []
        for filename in os.listdir(self.reports_dir):
            if filename.endswith('.csv') or filename.endswith('.txt'):
                files.append(os.path.join(self.reports_dir, filename))
        
        return sorted(files)
    
    def cleanup_old_reports(self, days_keep: int = 30):
        """Remove reports older than N days"""
        cutoff = datetime.now() - timedelta(days=days_keep)
        
        for filename in os.listdir(self.reports_dir):
            filepath = os.path.join(self.reports_dir, filename)
            
            try:
                mtime = os.path.getmtime(filepath)
                mtime_dt = datetime.fromtimestamp(mtime)
                
                if mtime_dt < cutoff:
                    os.remove(filepath)
            except Exception as e:
                print(f"Error removing {filepath}: {e}")
