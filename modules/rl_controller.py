import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MultiHeadRLController(nn.Module):
    def __init__(self, encoder_choices, pooling_choices, readout_choices,
                 augment_choices, hidden_dims, dropouts, lrs, bn_onoff, temp_choices):
        super().__init__()
        self.encoder_choices = encoder_choices
        self.pooling_choices = pooling_choices
        self.readout_choices = readout_choices
        self.augment_choices = augment_choices
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.lrs = lrs
        self.bn_onoff = bn_onoff
        self.temp_choices = temp_choices

        self.encoder_head = nn.Linear(32, len(encoder_choices))
        self.pooling_head = nn.Linear(32, len(pooling_choices))
        self.readout_head = nn.Linear(32, len(readout_choices))
        self.augment_head = nn.Linear(32, len(augment_choices))
        self.hidden_dim_head = nn.Linear(32, len(hidden_dims))
        self.dropout_head = nn.Linear(32, len(dropouts))
        self.lr_head = nn.Linear(32, len(lrs))
        self.bn_head = nn.Linear(32, len(bn_onoff))
        self.temp_head = nn.Linear(32, len(temp_choices))

        self.state_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

    def forward(self, state):
        s_emb = self.state_embed(state)
        logits_enc = self.encoder_head(s_emb)
        logits_pool = self.pooling_head(s_emb)
        logits_read = self.readout_head(s_emb)
        logits_aug = self.augment_head(s_emb)
        logits_hid = self.hidden_dim_head(s_emb)
        logits_drop = self.dropout_head(s_emb)
        logits_lr = self.lr_head(s_emb)
        logits_bn = self.bn_head(s_emb)
        logits_temp = self.temp_head(s_emb)
        return [logits_enc, logits_pool, logits_read, logits_aug,
                logits_hid, logits_drop, logits_lr, logits_bn, logits_temp]

    def sample_actions(self, state):
        all_logits = self.forward(state)
        actions = {}
        log_probs = []

        names = ["encoder", "pooling", "readout", "augment",
                 "hidden_dim", "dropout", "lr", "bn", "temperature"]
        choice_lists = [self.encoder_choices, self.pooling_choices, self.readout_choices,
                        self.augment_choices, self.hidden_dims, self.dropouts, self.lrs,
                        self.bn_onoff, self.temp_choices]

        for name, logits, clist in zip(names, all_logits, choice_lists):
            probs = F.softmax(logits, dim=-1)
            probs_np = probs[0].detach().cpu().numpy()
            idx = random.choices(range(len(clist)), weights=probs_np, k=1)[0]
            actions[name] = idx
            log_p = torch.log(probs[0, idx] + 1e-8)
            log_probs.append(log_p)

        total_log_prob = sum(log_probs)
        return actions, total_log_prob

    def parse_actions(self, actions):
        return {
            "encoder": self.encoder_choices[actions["encoder"]],
            "pooling": self.pooling_choices[actions["pooling"]],
            "readout": self.readout_choices[actions["readout"]],
            "augment": self.augment_choices[actions["augment"]],
            "hidden_dim": self.hidden_dims[actions["hidden_dim"]],
            "dropout": self.dropouts[actions["dropout"]],
            "lr": self.lrs[actions["lr"]],
            "bn": self.bn_onoff[actions["bn"]],
            "temperature": self.temp_choices[actions["temperature"]]
        }

    def reinforce_loss(self, log_prob, reward):
        return -log_prob * reward
