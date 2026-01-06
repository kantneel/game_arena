#!/usr/bin/env python3
"""Service for running matches in the background."""

import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class MatchConfig(BaseModel):
    """Configuration for starting a new match."""
    model_a: str
    model_b: str
    initial_time_seconds: int = 300
    increment_seconds: int = 3
    first_to: int = 1
    use_rethinking: bool = True
    max_rethinks: int = 2
    max_parsing_failures: int = 3
    # Per-model reasoning configuration
    reasoning_budget_a: int = 8000
    reasoning_budget_b: int = 8000
    show_reasoning_a: bool = False
    show_reasoning_b: bool = False
    # Optional notes/tags for experiment tracking
    notes: Optional[str] = None


@dataclass
class ProcessInfo:
    """Information about a running match process."""
    process: subprocess.Popen
    config: MatchConfig
    started_at: datetime
    logs: deque = field(default_factory=lambda: deque(maxlen=100))
    status: str = "starting"
    error: Optional[str] = None


class MatchRunner:
    """Manages running matches in background processes."""
    
    def __init__(self):
        self.processes: dict[int, ProcessInfo] = {}  # pid -> ProcessInfo
        self._project_root = Path(__file__).parent.parent.parent.parent
        self._lock = threading.Lock()
    
    def start_match(self, config: MatchConfig) -> dict:
        """Start a new match in a background process.
        
        Returns:
            dict with status and process info
        """
        # Build the command to run the match
        cmd = [
            sys.executable, "-m", "game_arena.blitz.match",
            f"--model_a={config.model_a}",
            f"--model_b={config.model_b}",
            f"--initial_time_seconds={config.initial_time_seconds}",
            f"--increment_seconds={config.increment_seconds}",
            f"--first_to={config.first_to}",
            f"--use_rethinking={str(config.use_rethinking).lower()}",
            f"--max_rethinks={config.max_rethinks}",
            f"--max_parsing_failures={config.max_parsing_failures}",
            # Per-model reasoning budgets
            f"--reasoning_budget_a={config.reasoning_budget_a}",
            f"--reasoning_budget_b={config.reasoning_budget_b}",
        ]
        
        # Add show reasoning traces if either model has it enabled
        if config.show_reasoning_a or config.show_reasoning_b:
            cmd.append("--show_reasoning_traces=true")
        
        # Add notes if provided
        if config.notes:
            cmd.append(f"--notes={config.notes}")
        
        try:
            # Set environment to force unbuffered output and colors
            import os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FORCE_COLOR"] = "1"  # Force colored output
            env["TERM"] = "xterm-256color"  # Pretend we have a color terminal
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )
            
            # Create process info
            info = ProcessInfo(
                process=process,
                config=config,
                started_at=datetime.now(),
                status="running",
            )
            
            with self._lock:
                self.processes[process.pid] = info
            
            # Start a thread to read output
            thread = threading.Thread(
                target=self._read_output,
                args=(process.pid,),
                daemon=True,
            )
            thread.start()
            
            # Log the command for debugging
            print(f"[MatchRunner] Starting match with command:")
            print(f"[MatchRunner] {' '.join(cmd)}")
            print(f"[MatchRunner] CWD: {self._project_root}")
            print(f"[MatchRunner] PID: {process.pid}")
            
            return {
                "status": "started",
                "process_id": process.pid,
                "message": f"Match started: {config.model_a} vs {config.model_b}",
                "command": " ".join(cmd),
                "cwd": str(self._project_root),
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to start match: {e}",
            }
    
    def _read_output(self, pid: int):
        """Read output from a process in a background thread."""
        with self._lock:
            info = self.processes.get(pid)
        
        if not info:
            return
        
        try:
            # Add initial log entry
            with self._lock:
                if pid in self.processes:
                    self.processes[pid].logs.append({
                        "time": datetime.now().isoformat(),
                        "line": f"[RUNNER] Process started with PID {pid}",
                    })
                    self.processes[pid].status = "running"
            
            # Read output line by line
            while True:
                line = info.process.stdout.readline()
                if not line:
                    # No more output - check if process is done
                    if info.process.poll() is not None:
                        break
                    continue
                
                line = line.rstrip()
                if line:  # Skip empty lines
                    with self._lock:
                        if pid in self.processes:
                            self.processes[pid].logs.append({
                                "time": datetime.now().isoformat(),
                                "line": line,
                            })
            
            # Process finished
            exit_code = info.process.poll()
            with self._lock:
                if pid in self.processes:
                    self.processes[pid].logs.append({
                        "time": datetime.now().isoformat(),
                        "line": f"[RUNNER] Process exited with code {exit_code}",
                    })
                    if exit_code == 0:
                        self.processes[pid].status = "completed"
                    else:
                        self.processes[pid].status = "failed"
                        self.processes[pid].error = f"Exit code: {exit_code}"
                        
        except Exception as e:
            with self._lock:
                if pid in self.processes:
                    self.processes[pid].logs.append({
                        "time": datetime.now().isoformat(),
                        "line": f"[RUNNER ERROR] {str(e)}",
                    })
                    self.processes[pid].status = "error"
                    self.processes[pid].error = str(e)
    
    def get_process_status(self, pid: int) -> Optional[dict]:
        """Get detailed status of a specific process."""
        with self._lock:
            info = self.processes.get(pid)
        
        if not info:
            return None
        
        # Check if still running
        poll = info.process.poll()
        
        return {
            "pid": pid,
            "status": info.status if poll is None else ("completed" if poll == 0 else "failed"),
            "exit_code": poll,
            "model_a": info.config.model_a,
            "model_b": info.config.model_b,
            "started_at": info.started_at.isoformat(),
            "running_seconds": (datetime.now() - info.started_at).total_seconds(),
            "logs": list(info.logs),
            "error": info.error,
        }
    
    def get_all_processes(self) -> list[dict]:
        """Get list of all tracked processes (running and recent)."""
        result = []
        
        with self._lock:
            for pid, info in list(self.processes.items()):
                poll = info.process.poll()
                
                status = info.status
                if poll is not None:
                    status = "completed" if poll == 0 else "failed"
                
                result.append({
                    "pid": pid,
                    "status": status,
                    "exit_code": poll,
                    "model_a": info.config.model_a,
                    "model_b": info.config.model_b,
                    "started_at": info.started_at.isoformat(),
                    "running_seconds": (datetime.now() - info.started_at).total_seconds(),
                    "log_count": len(info.logs),
                    "last_log": info.logs[-1]["line"] if info.logs else None,
                    "error": info.error,
                })
        
        return sorted(result, key=lambda x: x["started_at"], reverse=True)
    
    def stop_match(self, pid: int) -> bool:
        """Stop a running match by process ID."""
        with self._lock:
            info = self.processes.get(pid)
        
        if not info:
            return False
        
        try:
            info.process.terminate()
            info.status = "stopped"
            return True
        except Exception:
            return False
    
    def clear_finished(self) -> int:
        """Remove finished processes from tracking. Returns count removed."""
        removed = 0
        with self._lock:
            for pid in list(self.processes.keys()):
                if self.processes[pid].process.poll() is not None:
                    del self.processes[pid]
                    removed += 1
        return removed


# Global instance
_match_runner: Optional[MatchRunner] = None


def get_match_runner() -> MatchRunner:
    """Get the global match runner instance."""
    global _match_runner
    if _match_runner is None:
        _match_runner = MatchRunner()
    return _match_runner

