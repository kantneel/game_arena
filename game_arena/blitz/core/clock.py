#!/usr/bin/env python3
"""Player clock management for blitz chess matches."""

import dataclasses
import time
from typing import Optional


@dataclasses.dataclass
class PlayerClock:
    """Tracks time remaining for a player."""
    time_remaining: float  # seconds
    is_active: bool = False
    move_start_time: Optional[float] = None
    total_moves: int = 0
    total_thinking_time: float = 0.0
    
    def start_move(self):
        """Start timing a move."""
        self.is_active = True
        self.move_start_time = time.time()
    
    def end_move(self, network_latency: float = 0.0, increment_seconds: float = 0.0) -> float:
        """End timing a move and return actual thinking time."""
        if not self.is_active or self.move_start_time is None:
            return 0.0
        
        move_end_time = time.time()
        total_move_time = move_end_time - self.move_start_time
        thinking_time = max(0.0, total_move_time - network_latency)
        
        self.time_remaining -= thinking_time
        self.time_remaining += increment_seconds  # Add increment
        self.total_thinking_time += thinking_time
        self.total_moves += 1
        self.is_active = False
        self.move_start_time = None
        
        return thinking_time

