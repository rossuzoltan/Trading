import os

target_file = r'c:\dev\trading\train_agent.py'
if not os.path.exists(target_file):
    print("File not found")
    exit(1)

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# EnhancedLoggingCallback start around line 1086 (index 1085)
# It ends around line 1226 (index 1225)
# Looking at the surrounding code to verify anchors
start_anchor = "class EnhancedLoggingCallback(BaseCallback):"
end_anchor = "class TrainingDiagnosticsCallback(BaseCallback):"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_anchor in line and start_idx == -1:
        start_idx = i
    if end_anchor in line:
        end_idx = i
        break

if start_idx == -1 or end_idx == -1:
    print(f"Could not find anchors: start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

print(f"Found anchors: lines {start_idx+1} to {end_idx+1}")

replacement_code = """class EnhancedLoggingCallback(BaseCallback):
    \"\"\"
    Captures high-fidelity metrics from the environment's info dicts during rollouts.
    Optimized for high SPS by avoiding dictionary overhead in the step loop.
    \"\"\"
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._audit_logger = logging.getLogger("train_agent.audit")
        # Use lists for fast appending; converting to numpy at rollout end is efficient
        self.pnl_buffer: list[float] = []
        self.bonus_buffer: list[float] = []
        self.penalty_buffer: list[float] = []
        self.action_type_buffer: list[str] = []
        # Support system resource telemetry
        self.cpu_usage_buffer: list[float] = []
        self.gpu_usage_buffer: list[float] = []
        self.ram_usage_buffer: list[float] = []

    def _on_step(self) -> bool:
        # Collect metrics from all vector environment workers
        infos = self.locals.get("infos")
        if infos:
            # Support both RuntimeGymEnv and ForexTradingEnv info dict formats
            for info in infos:
                # 1. Rewards: use simplified top-level keys now shared by both env types
                self.pnl_buffer.append(float(info.get("reward_pnl", 0.0)))
                self.bonus_buffer.append(float(info.get("reward_bonus", 0.0)))
                self.penalty_buffer.append(float(info.get("reward_penalty", 0.0)))

                # 2. Actions: Map "OPEN" + direction -> "LONG"/"SHORT"
                act_type = info.get("action_type", info.get("selected_action_type", "HOLD"))
                if act_type == "OPEN":
                    direction = int(info.get("selected_action_direction", 0))
                    act = "LONG" if direction > 0 else "SHORT"
                else:
                    act = act_type
                self.action_type_buffer.append(str(act))

            # Capture system metrics at each step (sampled at monitor's interval)
            sys_metrics = resource_monitor.get_latest()
            self.cpu_usage_buffer.append(sys_metrics.cpu_pct)
            if sys_metrics.gpu_pct is not None:
                self.gpu_usage_buffer.append(sys_metrics.gpu_pct)
            self.ram_usage_buffer.append(sys_metrics.ram_pct)
        return True

    def _on_rollout_end(self) -> None:
        if not self.pnl_buffer:
            return

        total_steps = len(self.pnl_buffer)
        
        # Convert to numpy for fast vectorized math
        pnls = np.array(self.pnl_buffer, dtype=np.float32)
        bonuses = np.array(self.bonus_buffer, dtype=np.float32)
        penalties = np.array(self.penalty_buffer, dtype=np.float32)
        
        # 1. Action Distribution (Count using np.unique for speed)
        unique_actions, counts = np.unique(self.action_type_buffer, return_counts=True)
        action_counts = dict(zip(unique_actions, counts))
        
        standard_actions = ["HOLD", "LONG", "SHORT", "CLOSE"]
        audit_metrics = {}
        for act in standard_actions:
            count = int(action_counts.get(act, 0))
            audit_metrics[f"action_count_{act.lower()}"] = count
            audit_metrics[f"action_pct_{act.lower()}"] = float(count / total_steps)
            
            # Record to SB3 logger (for TensorBoard/Stdout)
            self.logger.record(f"rollout/action_count_{act.lower()}", count)
            self.logger.record(f"rollout/action_pct_{act.lower()}", count / total_steps)

        # 2. Reward Component Breakdown
        avg_pnl = float(np.mean(pnls))
        avg_bonus = float(np.mean(bonuses))
        avg_penalty = float(np.mean(penalties))
        
        audit_metrics.update({
            "reward_pnl_mean": avg_pnl,
            "reward_bonus_mean": avg_bonus,
            "reward_penalty_mean": avg_penalty,
        })
        
        self.logger.record("rollout/reward_pnl_mean", avg_pnl)
        self.logger.record("rollout/reward_bonus_mean", avg_bonus)
        self.logger.record("rollout/reward_penalty_mean", avg_penalty)

        # 3. Directional Indicators
        profitable_steps = int(np.sum(pnls > 0))
        win_rate = float(profitable_steps / total_steps)
        audit_metrics["step_win_rate"] = win_rate
        self.logger.record("rollout/step_win_rate", win_rate)

        # Include system metrics in audit
        if self.cpu_usage_buffer:
            audit_metrics.update({
                "system_cpu_pct_mean": float(np.mean(self.cpu_usage_buffer)),
                "system_ram_pct_mean": float(np.mean(self.ram_usage_buffer)),
            })
            if self.gpu_usage_buffer:
                audit_metrics["system_gpu_pct_mean"] = float(np.mean(self.gpu_usage_buffer))

        # Emit structured audit log
        self._audit_logger.info(
            f"Rollout Audit: Steps={total_steps} WinRate={win_rate:.1%} PnL={avg_pnl:.6f} Bonus={avg_bonus:.6f}",
            extra={
                "event": "rollout_audit",
                "rollout_steps": total_steps,
                **audit_metrics
            }
        )

        if self.verbose > 0:
            print(
                f"[Rollout Audit] PnL={avg_pnl:.6f} | Bonus={avg_bonus:.6f} | "
                f"Penalty={avg_penalty:.4f} | WinRate={win_rate:.1%}"
            )

        # Reset buffers
        self.pnl_buffer.clear()
        self.bonus_buffer.clear()
        self.penalty_buffer.clear()
        self.action_type_buffer.clear()
        self.cpu_usage_buffer.clear()
        self.gpu_usage_buffer.clear()
        self.ram_usage_buffer.clear()


"""

new_lines = lines[:start_idx] + [replacement_code + "\n"] + lines[end_idx:]

with open(target_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Repair completed successfully.")
