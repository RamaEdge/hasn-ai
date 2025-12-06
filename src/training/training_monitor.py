"""
Training Monitor and Dashboard for Automated Internet Training
Provides real-time monitoring and analysis of the training process
"""

import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

# Add path for brain imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingMonitor:
    """Monitors and analyzes automated training progress"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.metrics_history = deque(maxlen=1000)
        self.concept_timeline = defaultdict(list)
        self.quality_trends = deque(maxlen=500)

    def load_training_metrics(self) -> List[Dict[str, Any]]:
        """Load all available training metrics files"""
        metrics_files = []

        if not os.path.exists(self.output_dir):
            return metrics_files

        for filename in os.listdir(self.output_dir):
            if filename.startswith("automated_training_metrics_") and filename.endswith(".json"):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        data["filename"] = filename
                        data["filepath"] = filepath
                        metrics_files.append(data)
                except Exception as e:
                    print(f"⚠️  Error loading {filename}: {e}")

        # Sort by timestamp
        metrics_files.sort(key=lambda x: x.get("timestamp", ""))
        return metrics_files

    def analyze_learning_progress(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning progress over time"""
        if not metrics_data:
            return {"error": "No metrics data available"}

        analysis = {
            "total_sessions": len(metrics_data),
            "time_span": None,
            "learning_velocity": {},
            "concept_discovery_rate": [],
            "quality_trends": [],
            "most_effective_concepts": [],
            "learning_patterns": {},
        }

        # Calculate time span
        if len(metrics_data) > 1:
            first_time = datetime.fromisoformat(metrics_data[0]["timestamp"].replace("_", "T"))
            last_time = datetime.fromisoformat(metrics_data[-1]["timestamp"].replace("_", "T"))
            analysis["time_span"] = str(last_time - first_time)

        # Analyze learning velocity (concepts learned per hour)
        total_concepts = 0
        total_patterns = 0

        for i, data in enumerate(metrics_data):
            training_metrics = data.get("training_metrics", {})
            concepts_discovered = training_metrics.get("concepts_discovered", 0)
            patterns_learned = training_metrics.get("patterns_learned", 0)

            total_concepts += concepts_discovered
            total_patterns += patterns_learned

            # Track concept discovery rate
            analysis["concept_discovery_rate"].append(
                {
                    "session": i + 1,
                    "cumulative_concepts": total_concepts,
                    "new_concepts": concepts_discovered,
                }
            )

            # Track quality trends
            quality_scores = training_metrics.get("quality_scores", [])
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                analysis["quality_trends"].append(
                    {
                        "session": i + 1,
                        "average_quality": avg_quality,
                        "quality_variance": np.var(quality_scores),
                    }
                )

        # Calculate learning velocity
        if analysis["time_span"]:
            time_hours = self._parse_timedelta_hours(analysis["time_span"])
            if time_hours > 0:
                analysis["learning_velocity"] = {
                    "concepts_per_hour": total_concepts / time_hours,
                    "patterns_per_hour": total_patterns / time_hours,
                }

        # Find most effective concepts
        all_effectiveness = {}
        for data in metrics_data:
            effectiveness = data.get("pattern_effectiveness", {})
            for concept, info in effectiveness.items():
                if concept not in all_effectiveness:
                    all_effectiveness[concept] = []
                all_effectiveness[concept].append(info.get("effectiveness", 0))

        # Average effectiveness per concept
        concept_effectiveness = {
            concept: np.mean(scores) for concept, scores in all_effectiveness.items()
        }

        analysis["most_effective_concepts"] = sorted(
            concept_effectiveness.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return analysis

    def _parse_timedelta_hours(self, timedelta_str: str) -> float:
        """Parse timedelta string and return hours"""
        try:
            # Simple parsing for "X days, Y:Z:W" format
            if "day" in timedelta_str:
                parts = timedelta_str.split(", ")
                days = int(parts[0].split(" ")[0])
                time_part = parts[1] if len(parts) > 1 else "0:00:00"
            else:
                days = 0
                time_part = timedelta_str

            time_components = time_part.split(":")
            hours = int(time_components[0])
            minutes = int(time_components[1]) if len(time_components) > 1 else 0
            seconds = int(float(time_components[2])) if len(time_components) > 2 else 0

            total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
            return total_hours
        except:
            return 1.0  # Default fallback

    def generate_training_report(self) -> str:
        """Generate a comprehensive training report"""
        metrics_data = self.load_training_metrics()

        if not metrics_data:
            return "❌ No training data available. Start training first."

        analysis = self.analyze_learning_progress(metrics_data)

        report = []
        report.append("🧠 HASN Automated Training Report")
        report.append("=" * 50)
        report.append("")

        # Overview
        report.append("📊 Training Overview:")
        report.append(f"   Total Training Sessions: {analysis['total_sessions']}")
        report.append(f"   Training Time Span: {analysis.get('time_span', 'N/A')}")
        report.append("")

        # Latest metrics
        latest = metrics_data[-1]
        latest_metrics = latest.get("training_metrics", {})
        report.append("📈 Latest Session Metrics:")
        report.append(f"   Articles Collected: {latest_metrics.get('articles_collected', 0)}")
        report.append(f"   Patterns Learned: {latest_metrics.get('patterns_learned', 0)}")
        report.append(f"   Concepts Discovered: {latest_metrics.get('concepts_discovered', 0)}")

        if latest_metrics.get("quality_scores"):
            avg_quality = np.mean(latest_metrics["quality_scores"])
            report.append(f"   Average Quality Score: {avg_quality:.3f}")
        report.append("")

        # Learning velocity
        if analysis["learning_velocity"]:
            velocity = analysis["learning_velocity"]
            report.append("⚡ Learning Velocity:")
            report.append(f"   Concepts per Hour: {velocity['concepts_per_hour']:.2f}")
            report.append(f"   Patterns per Hour: {velocity['patterns_per_hour']:.2f}")
            report.append("")

        # Most effective concepts
        if analysis["most_effective_concepts"]:
            report.append("🎯 Most Effective Concepts:")
            for concept, effectiveness in analysis["most_effective_concepts"][:5]:
                report.append(f"   • {concept}: {effectiveness:.3f}")
            report.append("")

        # Recent concepts learned
        latest_concepts = latest.get("learned_concepts", [])
        if latest_concepts:
            report.append("🧩 Recently Learned Concepts:")
            for concept in latest_concepts[-10:]:
                report.append(f"   • {concept}")
            report.append("")

        # Quality trends
        if len(analysis["quality_trends"]) > 1:
            recent_quality = analysis["quality_trends"][-5:]
            avg_recent_quality = np.mean([q["average_quality"] for q in recent_quality])
            report.append("📊 Quality Trends:")
            report.append(f"   Recent Average Quality: {avg_recent_quality:.3f}")

            # Quality improvement/degradation
            if len(recent_quality) >= 2:
                quality_change = (
                    recent_quality[-1]["average_quality"] - recent_quality[0]["average_quality"]
                )
                trend = (
                    "📈 Improving"
                    if quality_change > 0
                    else "📉 Declining" if quality_change < 0 else "➡️  Stable"
                )
                report.append(f"   Quality Trend: {trend} ({quality_change:+.3f})")
            report.append("")

        # Configuration info
        config = latest.get("config", {})
        if config:
            report.append("⚙️  Current Configuration:")
            report.append(
                f"   Max Articles per Session: {config.get('max_articles_per_session', 'N/A')}"
            )
            report.append(f"   Collection Interval: {config.get('collection_interval', 'N/A')}s")
            report.append(f"   Min Quality Score: {config.get('min_quality_score', 'N/A')}")
            report.append("")

        # Recommendations
        report.append("💡 Recommendations:")

        if analysis["quality_trends"]:
            recent_quality = analysis["quality_trends"][-1]["average_quality"]
            if recent_quality < 0.5:
                report.append("   • Consider increasing min_article_quality_score threshold")
            elif recent_quality > 0.8:
                report.append("   • Quality is excellent! Consider increasing collection frequency")

        if analysis["learning_velocity"].get("concepts_per_hour", 0) < 1:
            report.append("   • Learning velocity is low. Consider:")
            report.append("     - Reducing article collection interval")
            report.append("     - Increasing max_articles_per_session")
            report.append("     - Adding more diverse sources")

        if len(latest_concepts) > 50:
            report.append("   • Large number of concepts learned. Consider:")
            report.append("     - Implementing concept clustering")
            report.append("     - Adding concept hierarchy")

        report.append("")
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(report)

    def create_visualization(self, save_path: str = None):
        """Create training progress visualization"""
        try:
            import matplotlib.pyplot as plt

            metrics_data = self.load_training_metrics()
            if not metrics_data:
                print("❌ No data available for visualization")
                return

            analysis = self.analyze_learning_progress(metrics_data)

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("HASN Automated Training Progress", fontsize=16, fontweight="bold")

            # 1. Concept Discovery Over Time
            concept_data = analysis["concept_discovery_rate"]
            if concept_data:
                sessions = [d["session"] for d in concept_data]
                cumulative = [d["cumulative_concepts"] for d in concept_data]
                new_concepts = [d["new_concepts"] for d in concept_data]

                ax1.plot(sessions, cumulative, "b-", linewidth=2, label="Cumulative Concepts")
                ax1.bar(
                    sessions,
                    new_concepts,
                    alpha=0.6,
                    color="lightblue",
                    label="New Concepts",
                )
                ax1.set_title("Concept Discovery Over Time")
                ax1.set_xlabel("Training Session")
                ax1.set_ylabel("Number of Concepts")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. Quality Trends
            quality_data = analysis["quality_trends"]
            if quality_data:
                sessions = [d["session"] for d in quality_data]
                quality = [d["average_quality"] for d in quality_data]
                variance = [d["quality_variance"] for d in quality_data]

                ax2.plot(sessions, quality, "g-", linewidth=2, label="Average Quality")
                ax2.fill_between(
                    sessions,
                    [q - v for q, v in zip(quality, variance)],
                    [q + v for q, v in zip(quality, variance)],
                    alpha=0.3,
                    color="green",
                    label="Quality Variance",
                )
                ax2.set_title("Content Quality Trends")
                ax2.set_xlabel("Training Session")
                ax2.set_ylabel("Quality Score")
                ax2.set_ylim(0, 1)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Most Effective Concepts
            effective_concepts = analysis["most_effective_concepts"][:8]
            if effective_concepts:
                concepts, effectiveness = zip(*effective_concepts)
                concepts = [
                    c[:15] + "..." if len(c) > 15 else c for c in concepts
                ]  # Truncate long names

                bars = ax3.barh(range(len(concepts)), effectiveness, color="orange", alpha=0.7)
                ax3.set_yticks(range(len(concepts)))
                ax3.set_yticklabels(concepts)
                ax3.set_title("Most Effective Concepts")
                ax3.set_xlabel("Effectiveness Score")
                ax3.grid(True, alpha=0.3)

                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, effectiveness)):
                    ax3.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

            # 4. Learning Velocity Gauge
            velocity = analysis.get("learning_velocity", {})
            concepts_per_hour = velocity.get("concepts_per_hour", 0)
            patterns_per_hour = velocity.get("patterns_per_hour", 0)

            # Simple bar chart for velocity metrics
            metrics = ["Concepts/Hour", "Patterns/Hour"]
            values = [concepts_per_hour, patterns_per_hour]
            colors = ["purple", "red"]

            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_title("Learning Velocity")
            ax4.set_ylabel("Rate (per hour)")
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"📊 Visualization saved to: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")

    def real_time_monitor(self, refresh_interval: int = 30):
        """Display real-time training progress"""
        print("🔄 Real-time Training Monitor")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 50)

        try:
            while True:
                # Clear screen (works on most terminals)
                os.system("clear" if os.name == "posix" else "cls")

                # Display current time
                print(f" Monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)

                # Load and display latest metrics
                metrics_data = self.load_training_metrics()

                if metrics_data:
                    latest = metrics_data[-1]
                    latest_metrics = latest.get("training_metrics", {})

                    print("📊 Current Training Status:")
                    print(f"   Total Sessions: {len(metrics_data)}")
                    print(f"   Articles Collected: {latest_metrics.get('articles_collected', 0)}")
                    print(f"   Patterns Learned: {latest_metrics.get('patterns_learned', 0)}")
                    print(f"   Concepts Discovered: {latest_metrics.get('concepts_discovered', 0)}")

                    if latest_metrics.get("quality_scores"):
                        avg_quality = np.mean(latest_metrics["quality_scores"])
                        print(f"   Average Quality: {avg_quality:.3f}")

                    # Show recent concepts
                    concepts = latest.get("learned_concepts", [])
                    if concepts:
                        print(f"   Latest Concepts: {', '.join(concepts[-5:])}")

                    print()

                    # Show training activity indicator
                    last_update = latest.get("timestamp", "")
                    if last_update:
                        try:
                            last_time = datetime.fromisoformat(last_update.replace("_", "T"))
                            time_since = datetime.now() - last_time

                            if time_since.total_seconds() < 3600:  # Less than 1 hour
                                status = " ACTIVE"
                            elif time_since.total_seconds() < 7200:  # Less than 2 hours
                                status = " RECENT"
                            else:
                                status = " IDLE"

                            print(f"Training Status: {status}")
                            print(f"Last Update: {time_since} ago")
                        except:
                            print("Training Status:  UNKNOWN")
                else:
                    print("❌ No training data found")
                    print(
                        "   Start training with: python -m src.training.automated_internet_trainer"
                    )

                print(f"\n🔄 Refreshing in {refresh_interval} seconds...")
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n Monitoring stopped")


def main():
    """Main function for training monitor CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="HASN Automated Training Monitor")
    parser.add_argument("--output-dir", default="output", help="Output directory for training data")
    parser.add_argument("--report", action="store_true", help="Generate training report")
    parser.add_argument("--visualize", action="store_true", help="Create training visualization")
    parser.add_argument("--monitor", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--save-viz", help="Save visualization to file")
    parser.add_argument(
        "--refresh", type=int, default=30, help="Monitor refresh interval (seconds)"
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(args.output_dir)

    if args.report:
        report = monitor.generate_training_report()
        print(report)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{args.output_dir}/training_report_{timestamp}.txt"
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\n💾 Report saved to: {report_file}")

    if args.visualize:
        save_path = args.save_viz
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{args.output_dir}/training_visualization_{timestamp}.png"

        monitor.create_visualization(save_path)

    if args.monitor:
        monitor.real_time_monitor(args.refresh)

    if not any([args.report, args.visualize, args.monitor]):
        print("🧠 HASN Training Monitor")
        print("Use --help to see available options")
        print("\nQuick commands:")
        print("  --report          Generate training report")
        print("  --visualize       Create training charts")
        print("  --monitor         Start real-time monitoring")


if __name__ == "__main__":
    main()
