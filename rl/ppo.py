from dataclasses import astuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from babyai_agent import get_size
from babyai_env import Spaces


class PPO:
    def __init__(
        self,
        agent,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):

        self.agent = agent

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(agent.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        gradient_norm = 0
        accuracy = 0
        total_loss = 0

        assert self.ppo_epoch == 1
        for e in range(self.ppo_epoch):
            if self.agent.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                inputs = Spaces(  # noqa: F841
                    *torch.split(
                        obs_batch,
                        [
                            get_size(space)
                            for space in astuple(self.agent.base.observation_spaces)
                        ],
                        dim=-1,
                    )
                )

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    logits,
                ) = self.agent.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )
                encoding = torch.split(inputs.encoding.long(), 1, dim=-1)
                loss = sum(
                    [
                        F.cross_entropy(inp, target.flatten())
                        for inp, target in zip(logits, encoding)
                    ]
                )
                preds = torch.stack([inp.argmax(-1) for inp in logits], dim=-1)
                accuracy += (preds == inputs.encoding).float().mean()

                #
                # ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                # surr1 = ratio * adv_targ
                # surr2 = (
                #     torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                #     * adv_targ
                # )
                # action_loss = -torch.min(surr1, surr2).mean()
                #
                # if self.use_clipped_value_loss:
                #     value_pred_clipped = value_preds_batch + (
                #         values - value_preds_batch
                #     ).clamp(-self.clip_param, self.clip_param)
                #     value_losses = (values - return_batch).pow(2)
                #     value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                #     value_loss = (
                #         0.5 * torch.max(value_losses, value_losses_clipped).mean()
                #     )
                # else:
                #     value_loss = 0.5 * (return_batch - values).pow(2).mean()
                #
                self.optimizer.zero_grad()
                loss.backward()
                total_loss += loss
                # (
                #     value_loss * self.value_loss_coef
                #     + action_loss
                #     - dist_entropy * self.entropy_coef
                # ).backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                gradient_norm += (
                    sum(
                        p.grad.detach().data.norm(2).item() ** 2
                        for p in self.agent.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                self.optimizer.step()

                # value_loss_epoch += value_loss.item()
                # action_loss_epoch += action_loss.item()
                # dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        total_loss /= num_updates
        gradient_norm /= num_updates
        accuracy /= num_updates

        return accuracy, total_loss, gradient_norm
