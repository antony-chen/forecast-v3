"""168-hour direct MLP model for rolling forecast origins.

For each sample, the forward encoder reads the immediately preceding 168 hours
chronologically and the reverse encoder reads that same window from the most
recent hour to the oldest. The decoder then predicts the following 168 hours.
The architecture is unchanged from the fixed weekly-origin version; rolling is
implemented by the Main script's sample construction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# M2oE2 meta transform
# ============================================================

class MetaNet(nn.Module):
    def __init__(self, input_dim: int, xprime_dim: int, feat_dim: int = 1):
        super().__init__()
        hidden_dim = max(input_dim * xprime_dim, 8)
        self.input_dim = int(input_dim)
        self.xprime_dim = int(xprime_dim)
        self.layer1 = nn.Linear(int(feat_dim), hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim * xprime_dim)

    def forward(self, x_feat: torch.Tensor) -> torch.Tensor:
        batch = x_feat.size(0)
        out = F.gelu(self.layer1(x_feat))
        out = self.layer2(out)
        return out.view(batch, self.input_dim, self.xprime_dim)


class GatingNet(nn.Module):
    def __init__(self, gate_input_dim: int, num_experts: int, gate_hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_experts = int(num_experts)
        hidden = int(gate_hidden_dim or max(gate_input_dim, 8))
        if self.num_experts > 0:
            self.layer1 = nn.Linear(gate_input_dim, hidden)
            self.layer2 = nn.Linear(hidden, self.num_experts)

    def forward(
        self,
        gate_input: torch.Tensor,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
    ) -> Optional[torch.Tensor]:
        if self.num_experts == 0:
            return None
        logits = self.layer2(F.leaky_relu(self.layer1(gate_input), negative_slope=0.01))
        if epoch is None or top_k is None or epoch < warmup_epochs or top_k <= 0:
            return torch.softmax(logits, dim=-1)
        k = min(int(top_k), self.num_experts)
        _, top_idx = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top_idx, True)
        return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


class MetaTransformBlock(nn.Module):
    """Map scalar load x into x_prime using external-data meta experts."""

    def __init__(
        self,
        xprime_dim: int,
        input_dim: int = 1,
        n_externals: int = 0,
        expert_specs: Optional[Sequence[Dict]] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.xprime_dim = int(xprime_dim)
        self.n_externals = int(n_externals)
        self.expert_specs: List[Dict] = [
            {"name": str(spec["name"]), "indices": [int(i) for i in spec["indices"]]}
            for spec in (expert_specs or [])
        ]
        if not self.expert_specs and self.n_externals > 0:
            self.expert_specs = [
                {"name": f"external_{i}", "indices": [i]}
                for i in range(self.n_externals)
            ]

        self.num_experts = len(self.expert_specs)
        self.meta_experts = nn.ModuleList([
            MetaNet(self.input_dim, self.xprime_dim, feat_dim=len(spec["indices"]))
            for spec in self.expert_specs
        ])
        self.expert_norms = nn.ModuleList([
            nn.LayerNorm([self.input_dim, self.xprime_dim])
            for _ in self.expert_specs
        ])
        self.gating = GatingNet(
            gate_input_dim=self.input_dim + self.n_externals,
            num_experts=self.num_experts,
        )

        self.theta0 = nn.Parameter(torch.empty(1, self.input_dim, self.xprime_dim))
        nn.init.xavier_normal_(self.theta0)

    def forward_batch_seq(
        self,
        x_l_seq: torch.Tensor,
        x_ext_seq: torch.Tensor,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
    ) -> torch.Tensor:
        if x_l_seq.ndim != 3:
            raise ValueError(f"x_l_seq must be [B,T,input_dim], got {tuple(x_l_seq.shape)}")
        if x_ext_seq.ndim != 3:
            raise ValueError(f"x_ext_seq must be [B,T,K], got {tuple(x_ext_seq.shape)}")
        batch, steps, _ = x_l_seq.shape
        x_l_flat = x_l_seq.reshape(batch * steps, self.input_dim)

        if self.num_experts == 0:
            theta = self.theta0.expand(batch * steps, -1, -1)
        else:
            x_ext_flat = x_ext_seq.reshape(batch * steps, self.n_externals)
            expert_weights = []
            for spec, expert, norm in zip(self.expert_specs, self.meta_experts, self.expert_norms):
                feat = x_ext_flat[:, spec["indices"]]
                expert_weights.append(norm(expert(feat)))
            stacked = torch.stack(expert_weights, dim=1)
            gate_input = torch.cat([x_l_flat, x_ext_flat], dim=-1)
            gates = self.gating(
                gate_input,
                epoch=epoch,
                top_k=top_k,
                warmup_epochs=warmup_epochs,
            )
            dynamic = (stacked * gates.view(batch * steps, self.num_experts, 1, 1)).sum(dim=1)
            theta = dynamic + self.theta0

        x_prime = torch.bmm(x_l_flat.unsqueeze(1), theta).squeeze(1)
        return x_prime.view(batch, steps, self.xprime_dim)



# ============================================================
# Encoder and reverse-context booster
# ============================================================

class VariationalEncoderMeta(nn.Module):
    """Forward weekly encoder plus an optional shared reverse-day phase encoder.

    The encoder week is reshaped into seven 24-hour days. The same small reverse
    GRU encodes each day independently in H23 -> ... -> H00 order, so every
    forecast origin receives a fixed-length previous-week day representation.
    Fusion is performed later against the matching rolling decoder state.
    """

    VALID_MODES = {"forward", "phase_aligned_dual", "reverse_week_dual"}

    def __init__(
        self,
        xprime_dim: int,
        hidden_size: int,
        latent_size: int,
        encoder_mode: str = "forward",
        num_layers: int = 1,
        dropout: float = 0.1,
        reverse_hidden_size: int = 48,
        fusion_bottleneck: int = 32,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if encoder_mode not in self.VALID_MODES:
            raise ValueError(
                f"encoder_mode must be one of {sorted(self.VALID_MODES)}, got {encoder_mode!r}"
            )
        if reverse_hidden_size <= 0:
            raise ValueError("reverse_hidden_size must be positive")
        if fusion_bottleneck <= 0:
            raise ValueError("fusion_bottleneck must be positive")
        if not (0.0 < residual_scale <= 1.0):
            raise ValueError("residual_scale must be in (0, 1]")

        self.encoder_mode = encoder_mode
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.reverse_hidden_size = int(reverse_hidden_size)
        self.fusion_bottleneck = int(fusion_bottleneck)
        self.residual_scale = float(residual_scale)
        self.num_layers = int(num_layers)
        self.hours_per_day = 24
        gru_dropout = dropout if num_layers > 1 else 0.0

        # Construct all baseline modules first so their fixed-seed initialization
        # remains identical across A, B, and C.
        self.rnn_forward = nn.GRU(
            xprime_dim, hidden_size, num_layers,
            batch_first=True, dropout=gru_dropout,
        )
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

        if encoder_mode in {"phase_aligned_dual", "reverse_week_dual"}:
            self.rnn_reverse = nn.GRU(
                xprime_dim, reverse_hidden_size, num_layers,
                batch_first=True, dropout=gru_dropout,
            )
            # Independent encoder spaces are fused by concatenation, not subtraction.
            self.fusion_down = nn.Linear(
                latent_size + reverse_hidden_size, fusion_bottleneck
            )
            self.fusion_up = nn.Linear(fusion_bottleneck, latent_size)
            # Exact forward-model behavior at initialization.
            nn.init.zeros_(self.fusion_up.weight)
            nn.init.zeros_(self.fusion_up.bias)
        else:
            self.rnn_reverse = None
            self.fusion_down = None
            self.fusion_up = None

        self._last_residual_ratio = 0.0
        self._last_phase_feature_norm = 0.0
        self._last_phase_day_diversity = 0.0

    def booster_diagnostics(self) -> Dict[str, float]:
        return {
            "residual_ratio": float(self._last_residual_ratio),
            "phase_feature_norm": float(self._last_phase_feature_norm),
            "phase_day_diversity": float(self._last_phase_day_diversity),
        }

    def booster_strength(self) -> float:
        return float(self._last_residual_ratio)

    @staticmethod
    def _origin_to_day_indices(phase_indices, n_days: int) -> List[int]:
        if phase_indices is None:
            selected = [24 * i for i in range(n_days)]
        elif torch.is_tensor(phase_indices):
            selected = [int(v) for v in phase_indices.detach().cpu().tolist()]
        else:
            selected = [int(v) for v in phase_indices]
        if not selected:
            raise ValueError("phase_indices is empty")
        invalid = [v for v in selected if v < 0 or v % 24 != 0 or (v // 24) >= n_days]
        if invalid:
            raise ValueError(
                f"phase indices must be day starts 0,24,...,{24*(n_days-1)}, got {invalid}"
            )
        return [v // 24 for v in selected]

    def forward(
        self,
        x_l_seq: torch.Tensor,
        x_ext_seq: torch.Tensor,
        transform_block: MetaTransformBlock,
        phase_indices=None,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
    ):
        x_prime = transform_block.forward_batch_seq(
            x_l_seq, x_ext_seq,
            epoch=epoch, top_k=top_k, warmup_epochs=warmup_epochs,
        )
        _, h_forward = self.rnn_forward(x_prime)
        h_f = h_forward[-1]
        mu_z = self.mu_layer(h_f)
        logvar_z = self.logvar_layer(h_f)

        if self.encoder_mode == "forward":
            phase_features = None
            self._last_residual_ratio = 0.0
            self._last_phase_feature_norm = 0.0
            self._last_phase_day_diversity = 0.0
        else:
            batch, steps, feat_dim = x_prime.shape
            if steps % self.hours_per_day != 0:
                raise ValueError(
                    f"encoder length must be divisible by 24, got {steps}"
                )
            n_days = steps // self.hours_per_day
            day_indices = self._origin_to_day_indices(phase_indices, n_days)

            if self.encoder_mode == "phase_aligned_dual":
                # Original reverse-day ablation: each 24-hour day is encoded
                # independently in H23 -> ... -> H00 order.
                day_seq = x_prime.reshape(
                    batch, n_days, self.hours_per_day, feat_dim
                )
                reverse_days = torch.flip(day_seq, dims=[2]).reshape(
                    batch * n_days, self.hours_per_day, feat_dim
                )
                _, h_reverse = self.rnn_reverse(reverse_days)
                phase_all = h_reverse[-1].reshape(
                    batch, n_days, self.reverse_hidden_size
                )
                phase_features = phase_all[:, day_indices, :]
            else:
                # Reverse the complete 168-hour week: most recent hour -> oldest hour.
                reverse_week = torch.flip(x_prime, dims=[1])
                _, h_reverse = self.rnn_reverse(reverse_week)
                week_feature = h_reverse[-1]
                phase_features = week_feature.unsqueeze(1).expand(
                    -1, len(day_indices), -1
                ).contiguous()

            with torch.no_grad():
                eps = 1e-8
                phase_norm = phase_features.norm(p=2, dim=-1)
                phase_mean = phase_features.mean(dim=1, keepdim=True)
                diversity = (
                    (phase_features - phase_mean).norm(p=2, dim=-1)
                    / phase_mean.norm(p=2, dim=-1).clamp_min(eps)
                )
                self._last_phase_feature_norm = float(
                    phase_norm.mean().detach().cpu()
                )
                self._last_phase_day_diversity = float(
                    diversity.mean().detach().cpu()
                )
                # Updated when fuse_phase is called in the decoder.
                self._last_residual_ratio = 0.0

        return mu_z, logvar_z, phase_features

    def fuse_phase(
        self,
        decoder_hidden: torch.Tensor,
        phase_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse independent day-phase features with matching decoder states."""
        if self.encoder_mode not in {"phase_aligned_dual", "reverse_week_dual"}:
            return decoder_hidden
        if decoder_hidden.ndim != 3 or phase_features.ndim != 3:
            raise ValueError("decoder_hidden and phase_features must both be [B,O,D]")
        if decoder_hidden.shape[:2] != phase_features.shape[:2]:
            raise ValueError(
                "decoder_hidden and phase_features must share batch/origin dimensions"
            )
        if decoder_hidden.size(-1) != self.latent_size:
            raise ValueError(
                f"decoder hidden size must be {self.latent_size}, got {decoder_hidden.size(-1)}"
            )
        if phase_features.size(-1) != self.reverse_hidden_size:
            raise ValueError(
                f"reverse-context feature size must be {self.reverse_hidden_size}, "
                f"got {phase_features.size(-1)}"
            )

        fusion_input = torch.cat([decoder_hidden, phase_features], dim=-1)
        delta_raw = self.fusion_up(F.gelu(self.fusion_down(fusion_input)))
        residual = self.residual_scale * delta_raw
        boosted = decoder_hidden + residual

        with torch.no_grad():
            eps = 1e-8
            hidden_norm = decoder_hidden.norm(p=2, dim=-1).clamp_min(eps)
            self._last_residual_ratio = float(
                (residual.norm(p=2, dim=-1) / hidden_norm).mean().detach().cpu()
            )
        return boosted


# ============================================================
# Rolling decoder + direct 24h MLP
# ============================================================

class VariationalDecoderDirectMLP(nn.Module):
    def __init__(
        self,
        xprime_dim: int,
        latent_size: int,
        output_len: int,
        output_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = -3.8,
        mlp_hidden: int = 70,
    ):
        super().__init__()
        self.latent_size = int(latent_size)
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.rnn = nn.GRU(
            xprime_dim,
            latent_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.horizon_mlp = nn.Sequential(
            nn.Linear(latent_size + output_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_mu = nn.Linear(mlp_hidden, output_len * output_dim)
        self.head_logvar = nn.Linear(mlp_hidden, output_len * output_dim)
        self.project = nn.ModuleList([
            nn.Linear(latent_size, latent_size) for _ in range(num_layers)
        ])

    def _decode_direct(self, hidden: torch.Tensor, go_value: torch.Tensor):
        trunk = self.horizon_mlp(torch.cat([hidden, go_value], dim=-1))
        mu = self.head_mu(trunk).view(-1, self.output_len, self.output_dim)
        raw_lv = self.head_logvar(trunk).view(-1, self.output_len, self.output_dim)
        logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw_lv)
        return mu, logvar

    @staticmethod
    def _origins(forecast_indices, max_origin: int) -> List[int]:
        if forecast_indices is None:
            selected = list(range(max_origin + 1))
        elif torch.is_tensor(forecast_indices):
            selected = [int(v) for v in forecast_indices.detach().cpu().tolist()]
        else:
            selected = [int(v) for v in forecast_indices]
        if not selected:
            raise ValueError("forecast_indices is empty")
        invalid = [v for v in selected if v < 0 or v > max_origin]
        if invalid:
            raise ValueError(f"forecast_indices must be in [0,{max_origin}], got {invalid}")
        return selected

    def forward(
        self,
        x_l_seq: torch.Tensor,
        x_ext_seq: torch.Tensor,
        z_latent: torch.Tensor,
        transform_block: MetaTransformBlock,
        initial_go: torch.Tensor,
        phase_features: Optional[torch.Tensor] = None,
        phase_fuser=None,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
        forecast_indices=None,
    ):
        batch, steps, _ = x_l_seq.shape
        h_init = torch.stack([self.project[i](z_latent) for i in range(self.num_layers)], dim=0)
        h0 = h_init[-1]

        x_prime = transform_block.forward_batch_seq(
            x_l_seq,
            x_ext_seq,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )
        out_seq, _ = self.rnn(x_prime, h_init)

        selected = self._origins(forecast_indices, steps)
        hidden_list, go_list = [], []
        if initial_go.ndim == 1:
            initial_go = initial_go.unsqueeze(-1)

        for origin in selected:
            if origin == 0:
                hidden_list.append(h0)
                go_list.append(initial_go)
            else:
                hidden_list.append(out_seq[:, origin - 1, :])
                go_list.append(x_l_seq[:, origin - 1, :])

        hidden = torch.stack(hidden_list, dim=1)
        go = torch.stack(go_list, dim=1)

        if phase_features is not None:
            if phase_fuser is None:
                raise ValueError("phase_fuser is required when phase_features are provided")
            if phase_features.shape[:2] != (batch, len(selected)):
                raise ValueError(
                    "phase_features must match the decoder batch and origin dimensions"
                )
            hidden = phase_fuser(hidden, phase_features)

        n_origins = hidden.size(1)
        mu_flat, lv_flat = self._decode_direct(
            hidden.reshape(batch * n_origins, self.latent_size),
            go.reshape(batch * n_origins, self.output_dim),
        )
        return (
            mu_flat.view(batch, n_origins, self.output_len, self.output_dim),
            lv_flat.view(batch, n_origins, self.output_len, self.output_dim),
        )


# ============================================================
# Full model
# ============================================================

class VariationalSeq2SeqMeta(nn.Module):
    def __init__(
        self,
        xprime_dim: int,
        input_dim: int,
        hidden_size: int,
        latent_size: int,
        output_len: int,
        n_externals: int,
        expert_specs: Optional[Sequence[Dict]],
        encoder_mode: str = "forward",
        output_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = -3.8,
        mlp_hidden: int = 70,
        reverse_hidden_size: int = 48,
        fusion_bottleneck: int = 32,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            input_dim=input_dim,
            n_externals=n_externals,
            expert_specs=expert_specs,
        )
        self.transform_dec = MetaTransformBlock(
            xprime_dim=xprime_dim,
            input_dim=input_dim,
            n_externals=n_externals,
            expert_specs=expert_specs,
        )
        self.decoder = VariationalDecoderDirectMLP(
            xprime_dim=xprime_dim,
            latent_size=latent_size,
            output_len=output_len,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            mlp_hidden=mlp_hidden,
        )
        self.encoder = VariationalEncoderMeta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            latent_size=latent_size,
            encoder_mode=encoder_mode,
            num_layers=num_layers,
            dropout=dropout,
            reverse_hidden_size=reverse_hidden_size,
            fusion_bottleneck=fusion_bottleneck,
            residual_scale=residual_scale,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        lv = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(
        self,
        enc_l: torch.Tensor,
        enc_ext: torch.Tensor,
        dec_l: torch.Tensor,
        dec_ext: torch.Tensor,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
        forecast_indices=None,
    ):
        selected = self.decoder._origins(forecast_indices, dec_l.size(1))
        mu_z, logvar_z, phase_features = self.encoder(
            enc_l,
            enc_ext,
            transform_block=self.transform_enc,
            phase_indices=selected,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )
        z_forward = self.reparameterize(mu_z, logvar_z)
        mu, logvar = self.decoder(
            dec_l,
            dec_ext,
            z_latent=z_forward,
            phase_features=phase_features,
            phase_fuser=self.encoder.fuse_phase if phase_features is not None else None,
            transform_block=self.transform_dec,
            initial_go=enc_l[:, -1, :],
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
            forecast_indices=selected,
        )
        return mu, logvar, mu_z, logvar_z


VariationalSeq2Seq_meta = VariationalSeq2SeqMeta


# ============================================================
# Direct168 v3: horizon-specific global head + local correction
# ============================================================
WEEK_HOURS = 168
DAY_STARTS = tuple(range(0, WEEK_HOURS, 24))


class Direct168GlobalLocalModel(nn.Module):
    def __init__(
        self,
        xprime_dim: int,
        input_dim: int,
        hidden_size: int,
        latent_size: int,
        n_encoder_externals: int,
        encoder_expert_specs: Optional[Sequence[Dict]],
        future_ext_dim: int,
        output_len: int = WEEK_HOURS,
        output_dim: int = 1,
        encoder_mode: str = "phase_aligned_dual",
        num_layers: int = 1,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = -3.8,
        reverse_hidden_size: int = 48,
        fusion_bottleneck: int = 32,
        residual_scale: float = 0.1,
        phase_summary_size: int = 32,
        future_summary_size: int = 32,
        global_hidden: int = 64,
        local_ext_hidden: int = 24,
        hourly_hidden: int = 64,
        local_correction_gain: float = 0.25,
    ):
        super().__init__()
        if output_len != WEEK_HOURS:
            raise ValueError(f"Direct168 requires output_len=168, got {output_len}")
        if output_dim != 1:
            raise ValueError("This implementation currently supports scalar load only")
        if local_correction_gain < 0:
            raise ValueError("local_correction_gain must be non-negative")

        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.latent_size = int(latent_size)
        self.encoder_mode = str(encoder_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.local_correction_gain = float(local_correction_gain)

        self.transform_enc = MetaTransformBlock(
            xprime_dim=xprime_dim,
            input_dim=input_dim,
            n_externals=n_encoder_externals,
            expert_specs=encoder_expert_specs,
        )
        self.encoder = VariationalEncoderMeta(
            xprime_dim=xprime_dim,
            hidden_size=hidden_size,
            latent_size=latent_size,
            encoder_mode=encoder_mode,
            num_layers=num_layers,
            dropout=dropout,
            reverse_hidden_size=reverse_hidden_size,
            fusion_bottleneck=fusion_bottleneck,
            residual_scale=residual_scale,
        )

        self.phase_summary = nn.GRU(
            latent_size,
            phase_summary_size,
            num_layers=1,
            batch_first=True,
        )
        self.future_summary = nn.GRU(
            future_ext_dim,
            future_summary_size,
            num_layers=1,
            batch_first=True,
        )

        context_dim = phase_summary_size + future_summary_size + latent_size + output_dim
        self.global_head = nn.Sequential(
            nn.Linear(context_dim, global_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, global_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Distinct weights for all 168 horizons preserve implicit position.
        self.global_mu = nn.Linear(global_hidden, output_len * output_dim)
        self.global_logvar = nn.Linear(global_hidden, output_len * output_dim)

        # Direct per-hour access to known future exogenous information.
        self.local_future = nn.Sequential(
            nn.Linear(future_ext_dim, local_ext_hidden),
            nn.GELU(),
            nn.LayerNorm(local_ext_hidden),
        )
        local_input_dim = global_hidden + local_ext_hidden + output_dim
        self.local_trunk = nn.Sequential(
            nn.Linear(local_input_dim, hourly_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hourly_hidden, hourly_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_delta_mu = nn.Linear(hourly_hidden, output_dim)

        # The correction starts at exactly zero. The global horizon-specific
        # branch therefore owns the initial prediction and local information is
        # learned only when it improves the objective.
        nn.init.zeros_(self.local_delta_mu.weight)
        nn.init.zeros_(self.local_delta_mu.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        lv = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def encode(
        self,
        enc_l: torch.Tensor,
        enc_ext: torch.Tensor,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
    ):
        return self.encoder(
            enc_l,
            enc_ext,
            transform_block=self.transform_enc,
            phase_indices=DAY_STARTS,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )

    def decode_from_z(
        self,
        z: torch.Tensor,
        phase_features: Optional[torch.Tensor],
        future_ext: torch.Tensor,
        previous_week_load: torch.Tensor,
    ):
        if previous_week_load.ndim != 3 or previous_week_load.shape[1] != self.output_len:
            raise ValueError(
                f"previous_week_load must be [batch,168,1], got {tuple(previous_week_load.shape)}"
            )
        if future_ext.ndim != 3 or future_ext.shape[1] != self.output_len:
            raise ValueError(
                f"future_ext must be [batch,168,K], got {tuple(future_ext.shape)}"
            )

        if phase_features is None:
            phase_states = z.unsqueeze(1).expand(-1, 7, -1)
        else:
            repeated_z = z.unsqueeze(1).expand(-1, phase_features.size(1), -1)
            phase_states = self.encoder.fuse_phase(repeated_z, phase_features)

        _, h_phase = self.phase_summary(phase_states)
        _, h_future = self.future_summary(future_ext)
        boundary_load = previous_week_load[:, -1, :]
        context = torch.cat([h_phase[-1], h_future[-1], z, boundary_load], dim=-1)
        global_state = self.global_head(context)

        base_mu = self.global_mu(global_state).view(
            -1, self.output_len, self.output_dim
        )
        raw_logvar = self.global_logvar(global_state).view(
            -1, self.output_len, self.output_dim
        )

        local_future = self.local_future(future_ext)
        global_hourly = global_state.unsqueeze(1).expand(-1, self.output_len, -1)
        local_input = torch.cat(
            [global_hourly, local_future, previous_week_load],
            dim=-1,
        )
        local_state = self.local_trunk(local_input)
        local_delta = self.local_delta_mu(local_state)

        mu = base_mu + self.local_correction_gain * local_delta
        logvar = self.logvar_min + (
            self.logvar_max - self.logvar_min
        ) * torch.sigmoid(raw_logvar)
        return mu, logvar

    def forward(
        self,
        enc_l: torch.Tensor,
        enc_ext: torch.Tensor,
        future_ext: torch.Tensor,
        epoch: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_epochs: int = 0,
        latent_mode: str = "sample",
    ):
        mu_z, logvar_z, phase_features = self.encode(
            enc_l,
            enc_ext,
            epoch=epoch,
            top_k=top_k,
            warmup_epochs=warmup_epochs,
        )
        if latent_mode == "mean":
            z = mu_z
        elif latent_mode == "sample":
            z = self.reparameterize(mu_z, logvar_z)
        else:
            raise ValueError(f"latent_mode must be 'sample' or 'mean', got {latent_mode!r}")

        mu, logvar = self.decode_from_z(
            z,
            phase_features,
            future_ext,
            previous_week_load=enc_l,
        )
        return mu, logvar, mu_z, logvar_z
