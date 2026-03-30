import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
from sb3_contrib import MaskablePPO
from torch.nn import functional as F
import logging

log = logging.getLogger("train_agent.amp")

class MaskablePPO_AMP(MaskablePPO):
    """
    Subclass of MaskablePPO that implements Automated Mixed Precision (AMP).
    Optimized for NVIDIA GPUs with Tensor Cores (RTX 30/40/50 series).
    Supports both BF16 (default) and FP16 with GradScaler.
    """

    def __init__(self, *args, amp_dtype: str = "bf16", **kwargs):
        super().__init__(*args, **kwargs)
        
        # Select device-compatible dtype
        if amp_dtype == "bf16":
            self.amp_dtype = th.bfloat16
        else:
            self.amp_dtype = th.float16
            
        # GradScaler is only needed for float16 to prevent underflow
        self.scaler = th.cuda.amp.GradScaler(enabled=(self.amp_dtype == th.float16))
        
        if self.verbose >= 1:
            log.info(f"[AMP] Initialized with dtype={self.amp_dtype} (scaler_enabled={self.scaler.is_enabled()})")

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer, with AMP support.
        Derived from sb3_contrib.MaskablePPO.train()
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # 1. Advantage normalization (Keep in FP32 for stability)
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- AMP Training Step ---
                self.policy.optimizer.zero_grad()

                # Selective autocast: Wrap forward pass and initial loss calculation
                with th.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )
                    values = values.flatten()

                # --- High Precision PPO Logic (FP32 Fallback) ---
                # We perform ratio, surrogate, and loss aggregation in FP32 
                # to prevent policy collapse from precision artifacts.
                with th.amp.autocast(device_type="cuda", enabled=False):
                    log_prob, values = log_prob.float(), values.float()
                    if entropy is not None:
                        entropy = entropy.float()

                    # 2. Ratio calculation (Sensitive exponential)
                    log_ratio = log_prob - rollout_data.old_log_prob
                    ratio = th.exp(log_ratio)

                    # 3. Clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # 4. Value loss
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)

                    # 5. Entropy loss (Exploration bonus)
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    # Total weighted loss (Aggregated in FP32)
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- Backward Pass with Scaler ---
                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping if using FP16
                self.scaler.unscale_(self.policy.optimizer)
                
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(self.policy.optimizer)
                self.scaler.update()

                # --- Metrics Accounting (FP32) ---
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                with th.no_grad():
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        log.info(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
