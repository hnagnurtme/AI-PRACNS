"""
Custom Training Dashboard for RL Training
Rich console output với health indicators và auto log cleanup
"""
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.style import Style
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

logger = logging.getLogger(__name__)


class TrainingHealthStatus:
    """Training health status indicators"""
    GOOD = "🟢"
    WARNING = "🟡"
    BAD = "🔴"
    UNKNOWN = "⚪"


class TrainingDashboard:
    """
    Custom Training Dashboard với:
    - Rich Console Output với màu sắc đẹp
    - Training Health Monitor
    - Auto Log Cleanup
    """
    
    def __init__(
        self,
        config: Dict = None,
        log_dir: Optional[Path] = None,
        auto_cleanup: bool = True,
        retention_days: int = 7,
        update_frequency: int = 10
    ):
        self.config = config or {}
        self.log_dir = log_dir
        self.auto_cleanup = auto_cleanup
        self.retention_days = retention_days
        self.update_frequency = update_frequency
        
        # Metrics history for trend analysis
        self.reward_history: Deque[float] = deque(maxlen=100)
        self.success_history: Deque[float] = deque(maxlen=100)
        self.loss_history: Deque[float] = deque(maxlen=100)
        self.epsilon_history: Deque[float] = deque(maxlen=100)
        
        # Training state
        self.start_time: Optional[datetime] = None
        self.current_episode = 0
        self.total_episodes = 0
        self.best_reward = float('-inf')
        
        # Console
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
            logger.warning("Rich library not available. Install with: pip install rich")
    
    def start(self, total_episodes: int):
        """Start training dashboard"""
        self.start_time = datetime.now()
        self.total_episodes = total_episodes
        
        if self.console:
            self._print_header()
    
    def _print_header(self):
        """Print training header with beautiful styling"""
        if not self.console:
            return
        
        header = Panel(
            Text.from_markup(
                "[bold cyan]🚀 RL TRAINING DASHBOARD[/bold cyan]\n"
                f"[dim]Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"
                f"[dim]Episodes: {self.total_episodes}[/dim]"
            ),
            title="[bold white]SAGIN Routing Agent[/bold white]",
            subtitle="[dim]Dueling DQN Training[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        )
        self.console.print(header)
        self.console.print()
    
    def update(
        self,
        episode: int,
        reward: float,
        success: bool,
        loss: float,
        epsilon: float,
        extra_metrics: Optional[Dict] = None
    ):
        """Update dashboard with new episode data"""
        self.current_episode = episode
        self.reward_history.append(reward)
        self.success_history.append(1.0 if success else 0.0)
        if loss > 0:
            self.loss_history.append(loss)
        self.epsilon_history.append(epsilon)
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        # Only display every update_frequency episodes
        if episode % self.update_frequency == 0:
            self._display_update(episode, extra_metrics)
    
    def _display_update(self, episode: int, extra_metrics: Optional[Dict] = None):
        """Display training update with health indicators"""
        if not self.console:
            return
        
        # Calculate metrics
        avg_reward = np.mean(list(self.reward_history)[-10:]) if self.reward_history else 0
        success_rate = np.mean(list(self.success_history)[-10:]) if self.success_history else 0
        avg_loss = np.mean(list(self.loss_history)[-10:]) if self.loss_history else 0
        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 1.0
        
        # Get health status
        reward_health = self._get_reward_health()
        success_health = self._get_success_health()
        loss_health = self._get_loss_health()
        
        # Overall health
        overall_health = self._get_overall_health(reward_health, success_health, loss_health)
        
        # Create progress percentage
        progress_pct = (episode / self.total_episodes) * 100
        
        # Create table
        table = Table(
            title=f"[bold]Episode {episode}/{self.total_episodes}[/bold] ({progress_pct:.1f}%)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", justify="right", width=12)
        table.add_column("Avg (10ep)", justify="right", width=12)
        table.add_column("Health", justify="center", width=8)
        
        # Add rows with color coding
        reward_style = self._get_style(reward_health)
        table.add_row(
            "Reward",
            f"{self.reward_history[-1]:.1f}" if self.reward_history else "N/A",
            f"{avg_reward:.1f}",
            f"{reward_health}"
        )
        
        success_style = self._get_style(success_health)
        table.add_row(
            "Success Rate",
            "✓" if self.success_history and self.success_history[-1] > 0 else "✗",
            f"{success_rate:.1%}",
            f"{success_health}"
        )
        
        loss_style = self._get_style(loss_health)
        table.add_row(
            "Loss",
            f"{self.loss_history[-1]:.4f}" if self.loss_history else "N/A",
            f"{avg_loss:.4f}",
            f"{loss_health}"
        )
        
        table.add_row(
            "Epsilon",
            f"{current_epsilon:.3f}",
            "",
            self._get_epsilon_health()
        )
        
        table.add_row(
            "Best Reward",
            f"{self.best_reward:.1f}",
            "",
            ""
        )
        
        # Print table with overall status
        self.console.print()
        overall_color = "green" if overall_health == "GOOD" else ("yellow" if overall_health == "WARNING" else "red")
        self.console.print(f"[bold {overall_color}]Training Status: {overall_health}[/bold {overall_color}]")
        self.console.print(table)
        
        # Print extra metrics if available
        if extra_metrics:
            self._print_extra_metrics(extra_metrics)
    
    def _print_extra_metrics(self, metrics: Dict):
        """Print additional metrics"""
        if not self.console or not metrics:
            return
        
        extra_table = Table(box=box.SIMPLE, show_header=False)
        extra_table.add_column("Metric", style="dim")
        extra_table.add_column("Value", justify="right")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                extra_table.add_row(key, f"{value:.2f}")
            else:
                extra_table.add_row(key, str(value))
        
        self.console.print(extra_table)
    
    def _get_reward_health(self) -> str:
        """Get reward trend health status"""
        if len(self.reward_history) < 5:
            return TrainingHealthStatus.UNKNOWN
        
        recent = list(self.reward_history)[-5:]
        older = list(self.reward_history)[-10:-5] if len(self.reward_history) >= 10 else recent[:len(recent)//2]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older) if older else recent_avg
        
        trend = recent_avg - older_avg
        
        if trend > 10:  # Improving significantly
            return TrainingHealthStatus.GOOD
        elif trend > -10:  # Stable or slight improvement
            return TrainingHealthStatus.WARNING
        else:  # Declining
            return TrainingHealthStatus.BAD
    
    def _get_success_health(self) -> str:
        """Get success rate health status"""
        if len(self.success_history) < 10:
            return TrainingHealthStatus.UNKNOWN
        
        success_rate = np.mean(list(self.success_history)[-10:])
        
        if success_rate >= 0.7:
            return TrainingHealthStatus.GOOD
        elif success_rate >= 0.4:
            return TrainingHealthStatus.WARNING
        else:
            return TrainingHealthStatus.BAD
    
    def _get_loss_health(self) -> str:
        """Get loss stability health status"""
        if len(self.loss_history) < 5:
            return TrainingHealthStatus.UNKNOWN
        
        recent = list(self.loss_history)[-10:]
        avg_loss = np.mean(recent)

        if avg_loss < 100:
            return TrainingHealthStatus.GOOD
        elif avg_loss < 500:
            return TrainingHealthStatus.WARNING
        else:
            return TrainingHealthStatus.BAD
    
    def _get_epsilon_health(self) -> str:
        """Get epsilon decay health"""
        if not self.epsilon_history:
            return TrainingHealthStatus.UNKNOWN
        
        current = self.epsilon_history[-1]
        expected_progress = self.current_episode / max(self.total_episodes, 1)
        
        # Epsilon should roughly follow: 1.0 -> 0.01 over training
        expected_epsilon = max(0.01, 1.0 - expected_progress * 0.99)
        
        if abs(current - expected_epsilon) < 0.2:
            return TrainingHealthStatus.GOOD
        elif current > expected_epsilon + 0.3:
            return TrainingHealthStatus.WARNING  # Too high
        else:
            return TrainingHealthStatus.WARNING
    
    def _get_overall_health(self, reward_h: str, success_h: str, loss_h: str) -> str:
        """Determine overall training health"""
        health_scores = {
            TrainingHealthStatus.GOOD: 2,
            TrainingHealthStatus.WARNING: 1,
            TrainingHealthStatus.BAD: 0,
            TrainingHealthStatus.UNKNOWN: 1
        }
        
        total = health_scores.get(reward_h, 1) + health_scores.get(success_h, 1) + health_scores.get(loss_h, 1)
        
        if total >= 5:
            return "GOOD"
        elif total >= 3:
            return "WARNING"
        else:
            return "BAD"
    
    def _get_style(self, health: str) -> str:
        """Get rich style based on health status"""
        if health == TrainingHealthStatus.GOOD:
            return "green"
        elif health == TrainingHealthStatus.WARNING:
            return "yellow"
        elif health == TrainingHealthStatus.BAD:
            return "red"
        return "dim"
    
    def show_final_summary(self, final_metrics: Optional[Dict] = None):
        """Show final training summary"""
        if not self.console:
            return
        
        duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        # Overall success determination
        final_success_rate = np.mean(list(self.success_history)[-50:]) if self.success_history else 0
        final_reward = np.mean(list(self.reward_history)[-50:]) if self.reward_history else 0
        
        if final_success_rate >= 0.7 and final_reward > 0:
            status_text = "[bold green]✅ TRAINING SUCCESSFUL[/bold green]"
            status_color = "green"
        elif final_success_rate >= 0.4:
            status_text = "[bold yellow]⚠️ TRAINING MODERATE[/bold yellow]"
            status_color = "yellow"
        else:
            status_text = "[bold red]❌ TRAINING NEEDS IMPROVEMENT[/bold red]"
            status_color = "red"
        
        # Create summary panel
        summary_text = Text()
        summary_text.append(f"\n{status_text}\n\n", style="bold")
        summary_text.append(f"Duration: {duration}\n")
        summary_text.append(f"Episodes: {self.current_episode}/{self.total_episodes}\n")
        summary_text.append(f"Best Reward: {self.best_reward:.1f}\n")
        summary_text.append(f"Final Success Rate: {final_success_rate:.1%}\n")
        summary_text.append(f"Final Avg Reward (50ep): {final_reward:.1f}\n")
        
        if final_metrics:
            summary_text.append("\nFinal Metrics:\n", style="bold")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    summary_text.append(f"  {key}: {value:.2f}\n")
                else:
                    summary_text.append(f"  {key}: {value}\n")
        
        panel = Panel(
            summary_text,
            title="[bold white]📊 TRAINING COMPLETE[/bold white]",
            border_style=status_color,
            box=box.DOUBLE
        )
        
        self.console.print()
        self.console.print(panel)
        
        # Auto cleanup logs
        if self.auto_cleanup:
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Cleanup old TensorBoard logs"""
        if not self.log_dir or not self.log_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cleaned_count = 0
        
        try:
            for item in self.log_dir.iterdir():
                if item.is_dir():
                    # Check modification time
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff_date:
                        shutil.rmtree(item)
                        cleaned_count += 1
            
            if cleaned_count > 0 and self.console:
                self.console.print(
                    f"[dim]🧹 Cleaned up {cleaned_count} old log directories "
                    f"(older than {self.retention_days} days)[/dim]"
                )
        except Exception as e:
            logger.warning(f"Error cleaning up logs: {e}")
    
    def print_simple_update(
        self,
        episode: int,
        reward: float,
        success: bool,
        loss: float,
        epsilon: float
    ):
        """Simple console update when rich is not available"""
        self.current_episode = episode
        self.reward_history.append(reward)
        self.success_history.append(1.0 if success else 0.0)
        if loss > 0:
            self.loss_history.append(loss)
        self.epsilon_history.append(epsilon)
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        if episode % self.update_frequency == 0:
            avg_reward = np.mean(list(self.reward_history)[-10:]) if self.reward_history else 0
            success_rate = np.mean(list(self.success_history)[-10:]) if self.success_history else 0
            
            print(
                f"📊 Ep {episode}/{self.total_episodes} | "
                f"Reward: {reward:.1f} (avg: {avg_reward:.1f}) | "
                f"Success: {success_rate:.1%} | "
                f"ε: {epsilon:.3f}"
            )
