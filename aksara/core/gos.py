"""
GOS - Generator Output Struktural (Autoregressive Edition)
Autoregressive BSU-level generation dengan Sequential State Evolution.

Pipeline:
  Encoding:  BSU states → predictors (root, affix, role, context)
  Generation: state_0 → Phi_seq → BSU_1 → Phi_seq → BSU_2 → ... → EOS

Phi_seq TERPISAH dari MEB:
  - MEB = understanding (f_morph + f_syn + f_sem)
  - Phi_seq = generation dynamics (morph_preserve + context_update + transition_bias)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from aksara.core.bsu import BSUConfig


@dataclass
class GOSConfig:
    bsu_config: BSUConfig = None
    vocab_size: int = 5000
    affix_vocab_size: int = 40
    role_vocab_size: int = 8
    dropout: float = 0.1
    use_ripl: bool = True
    # Autoregressive config
    max_gen_length: int = 50
    alpha_context: float = 0.7       # rolling context decay
    phi_seq_layers: int = 2          # layers in Phi_seq
    teacher_forcing: bool = True     # use teacher forcing during training

    def __post_init__(self):
        if self.bsu_config is None:
            self.bsu_config = BSUConfig()


class RIPL(nn.Module):
    """
    RIPL - Residual Informasi Preservasi Linguistik
    Gated skip connection yang mempertahankan informasi BSU original.

    gate = sigmoid(W_g * [h_evolved, h_original])
    output = gate * h_evolved + (1 - gate) * h_original
    """

    def __init__(self, d_total: int):
        super().__init__()
        self.gate = nn.Linear(d_total * 2, d_total)
        self.norm = nn.LayerNorm(d_total)

    def forward(self, h_evolved: torch.Tensor, h_original: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_evolved, h_original], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        out = g * h_evolved + (1 - g) * h_original
        return self.norm(out)


class SequentialStateEvolution(nn.Module):
    """
    Phi_seq - Generation Dynamics (TERPISAH dari MEB)

    Phi_seq(state_t, context_t) = morph_preserve + context_update + transition_bias

    Ini BUKAN MEB. MEB = understanding. Phi_seq = generation.
    - morph_preserve: jaga konsistensi morfologi antar step
    - context_update: update konteks berdasarkan state baru
    - transition_bias: bias transisi natural bahasa Indonesia

    Input:  state_t (d_total), context_t (d_total)
    Output: state_{t+1} (d_total)
    """

    def __init__(self, d_total: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_total = d_total

        # Morph preserve: jaga slot morfologi tetap konsisten
        self.morph_preserve = nn.Sequential(
            nn.Linear(d_total, d_total),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_total, d_total),
        )
        self.morph_gate = nn.Linear(d_total * 2, d_total)

        # Context update: integrasikan konteks rolling
        self.context_update = nn.Sequential(
            nn.Linear(d_total * 2, d_total),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_total, d_total),
        )
        self.context_gate = nn.Linear(d_total * 2, d_total)

        # Prompt anchor gate: tarik state kembali ke prompt setiap step
        # Ini yang memastikan generasi tidak drift dari makna prompt
        self.prompt_gate = nn.Linear(d_total * 2, d_total)
        self.prompt_proj  = nn.Linear(d_total, d_total)

        # Transition bias: model transisi natural antar BSU
        self.transition_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.transition_layers.append(nn.Sequential(
                nn.Linear(d_total, d_total * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_total * 2, d_total),
                nn.Dropout(dropout),
            ))

        self.norm_out = nn.LayerNorm(d_total)
        self.residual_gate = nn.Linear(d_total * 2, d_total)

    def forward(
        self,
        state_t: torch.Tensor,
        context_t: torch.Tensor,
        prompt_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_t       : (B, d_total) atau (B, 1, d_total) - state saat ini
            context_t     : (B, d_total) atau (B, 1, d_total) - rolling context
            prompt_anchor : (B, d_total) opsional - anchor dari prompt asli
                            Jika diberikan, state ditarik kembali ke ruang makna prompt
                            setiap step via learned gate. Ini memastikan generasi
                            tidak drift dari makna prompt.

        Returns:
            state_next : (B, d_total) atau (B, 1, d_total) - state berikutnya
        """
        squeeze = False
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(1)
            context_t = context_t.unsqueeze(1)
            if prompt_anchor is not None and prompt_anchor.dim() == 2:
                prompt_anchor = prompt_anchor.unsqueeze(1)
            squeeze = True

        # 1. Morph preserve
        morph_out = self.morph_preserve(state_t)
        mg = torch.sigmoid(self.morph_gate(torch.cat([morph_out, state_t], dim=-1)))
        morph_preserved = mg * morph_out + (1 - mg) * state_t

        # 2. Context update
        ctx_input = torch.cat([morph_preserved, context_t], dim=-1)
        ctx_out = self.context_update(ctx_input)
        cg = torch.sigmoid(self.context_gate(torch.cat([ctx_out, morph_preserved], dim=-1)))
        ctx_updated = cg * ctx_out + (1 - cg) * morph_preserved

        # 3. Transition bias (stacked layers with residual)
        h = ctx_updated
        for layer in self.transition_layers:
            delta = layer(h)
            h = h + delta  # residual

        # 4. Prompt anchor injection (kunci untuk prompt conditioning)
        # Gate belajar: seberapa kuat prompt harus menarik state saat ini
        # Tanpa ini: state hanya mengikuti distribusi global
        # Dengan ini: state selalu "ingat" makna prompt
        if prompt_anchor is not None:
            p_proj = self.prompt_proj(prompt_anchor)
            pg = torch.sigmoid(self.prompt_gate(torch.cat([h, p_proj], dim=-1)))
            h = h + pg * p_proj  # additive: prompt menambah sinyal, tidak mengganti

        # Final gated residual from original state
        rg = torch.sigmoid(self.residual_gate(torch.cat([h, state_t], dim=-1)))
        state_next = self.norm_out(rg * h + (1 - rg) * state_t)

        if squeeze:
            state_next = state_next.squeeze(1)

        return state_next


class GeneratorOutputStruktural(nn.Module):
    """
    GOS: Generator Output Struktural

    Encoding path (forward):
      MEB output → predictors → logits (root, affix, role, context)

    Generation path (generate):
      prompt → BSU_0 → Phi_seq → BSU_1 → ... → EOS
      Autoregressive BSU-level, bukan token-level.
    """

    def __init__(self, config: GOSConfig):
        super().__init__()
        self.config = config
        cfg = config.bsu_config
        d = cfg.d_total

        # RIPL: gated skip connection
        self.ripl = RIPL(d) if config.use_ripl else None

        # Predictors: dari BSU state → logits
        self.root_predictor = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, config.vocab_size),
        )

        self.affix_predictor = nn.Sequential(
            nn.Linear(d, cfg.d_morpheme),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(cfg.d_morpheme, config.affix_vocab_size),
        )

        self.role_predictor = nn.Sequential(
            nn.Linear(d, cfg.d_role),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(cfg.d_role, config.role_vocab_size),
        )

        self.context_predictor = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, config.vocab_size),
        )

        # Sequential State Evolution (Phi_seq) - TERPISAH dari MEB
        self.phi_seq = SequentialStateEvolution(
            d_total=d,
            n_layers=config.phi_seq_layers,
            dropout=config.dropout,
        )

        # State-to-BSU decoder: convert Phi_seq output back to BSU-compatible state
        self.state_to_bsu = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # EOS predictor: should we stop generating?
        self.eos_predictor = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
        )

        # Semantic bias projector: state → vocab space
        # Digunakan untuk semantic_bias = α * (state_proj · vocab_emb^T)
        # Ini yang membuat makna prompt eksplisit ikut dalam token decision
        self.semantic_bias_proj = nn.Linear(d, d, bias=False)

        # vocab_kbbi_mask: (V,) bool — True jika token ada di KBBI
        # Diisi oleh AksaraModel via set_kbbi_mask() setelah LSK dibangun
        self._vocab_kbbi_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        h_evolved: torch.Tensor,
        h_original: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encoding path: MEB output → prediction logits.

        Args:
            h_evolved  : (B, L, d_total) - output dari MEB
            h_original : (B, L, d_total) - BSU state original (untuk RIPL)

        Returns:
            dict with root_logits, affix_logits, role_logits, context_logits
        """
        # Apply RIPL if available
        if self.ripl is not None and h_original is not None:
            h = self.ripl(h_evolved, h_original)
        else:
            h = h_evolved

        root_logits = self.root_predictor(h)
        affix_logits = self.affix_predictor(h)
        role_logits = self.role_predictor(h)
        context_logits = self.context_predictor(h)

        return {
            "root_logits": root_logits,
            "affix_logits": affix_logits,
            "role_logits": role_logits,
            "context_logits": context_logits,
            "h_final": h,
        }

    def set_kbbi_mask(self, vocab_kbbi_mask: torch.Tensor):
        """
        Set vocab_kbbi_mask dari LSK — (V,) bool, True jika token ada di KBBI.
        Dipanggil oleh AksaraModel.__init__ setelah LSK dibangun.
        Digunakan untuk KBBI active boost di generate().
        """
        self._vocab_kbbi_mask = vocab_kbbi_mask

    def generate(
        self,
        prompt_state: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        eos_id: int = 3,
        min_length: int = 4,
        prompt_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation: BSU-level.

        state_0 = prompt_state (dari encoding)
        for t in range(max_length):
            context_t = alpha * context_{t-1} + (1-alpha) * state_t
            state_{t+1} = Phi_seq(state_t, context_t)
            word_{t+1} = decode(state_{t+1})
            if EOS: break

        Args:
            prompt_state : (B, d_total) atau (B, L, d_total) - initial state
            max_length   : maximum generation steps
            temperature  : sampling temperature
            eos_id       : EOS token id

        Returns:
            dict with generated_ids, generated_states, eos_probs
        """
        max_len = max_length or self.config.max_gen_length
        alpha = self.config.alpha_context

        # Handle prompt: take mean of all positions as initial state
        if prompt_state.dim() == 3:
            state_t = prompt_state.mean(dim=1)  # (B, d_total)
        else:
            state_t = prompt_state  # (B, d_total)

        # prompt_anchor = representasi tetap prompt, dipakai setiap step
        # Ini yang memastikan generasi selalu "ingat" makna prompt asli
        prompt_anchor = state_t.clone()  # (B, d_total)

        B = state_t.size(0)
        device = state_t.device

        # Initialize rolling context
        context_t = state_t.clone()

        # Storage
        generated_root_ids = []
        generated_affix_ids = []
        generated_states = []
        eos_probs = []
        active_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Ambil embedding matrix root_predictor layer terakhir
        # untuk semantic bias (state_proj · W_vocab^T)
        # Ini hubungkan semantik prompt ke ruang vocab secara eksplisit
        vocab_weight = self.root_predictor[-1].weight  # (V, d)

        for step in range(max_len):
            # 1. Evolve state via Phi_seq (dengan prompt anchor setiap step)
            state_next = self.phi_seq(state_t, context_t, prompt_anchor=prompt_anchor)

            # 2. Decode BSU → word
            h_decode = self.state_to_bsu(state_next)
            root_logits = self.root_predictor(h_decode.unsqueeze(1)).squeeze(1)  # (B, V)
            affix_logits = self.affix_predictor(h_decode.unsqueeze(1)).squeeze(1)  # (B, A)

            # 3a. SEMANTIC BIAS: sinyal makna prompt → logits
            # Alpha dikecilkan 0.3→0.15 agar tidak override language model prior
            prompt_proj = self.semantic_bias_proj(prompt_anchor)  # (B, d)
            semantic_bias = prompt_proj @ vocab_weight.t()  # (B, V)
            semantic_bias = 0.15 * F.normalize(semantic_bias, dim=-1)
            root_logits = root_logits + semantic_bias

            # 3b. KBBI ACTIVE BOOST: token KBBI mendapat +0.3 logit
            # Hanya token yang sudah di-generate (bukan prompt) yang di-nerf
            # agar model tidak diblok dari menghasilkan kata aksi domain
            if self._vocab_kbbi_mask is not None:
                kbbi_mask = self._vocab_kbbi_mask.to(root_logits.device).float().clone()
                if len(generated_root_ids) > 0:
                    # Nerf hanya token yang sudah muncul di output sebelumnya
                    recent = torch.stack(generated_root_ids[-3:], dim=1)  # (B, ≤3)
                    for b in range(B):
                        already_said = recent[b]
                        valid = already_said[
                            (already_said > 3) & (already_said < kbbi_mask.size(0))
                        ]
                        if valid.numel() > 0:
                            kbbi_mask[valid] = kbbi_mask[valid] * 0.3  # nerf yang sudah keluar
                root_logits = root_logits + kbbi_mask * 0.3

            # 3c. Suppress noise tokens
            root_logits_masked = root_logits.clone()
            root_logits_masked[:, 0] = float('-inf')  # PAD
            root_logits_masked[:, 1] = float('-inf')  # UNK
            root_logits_masked[:, 2] = float('-inf')  # BOS
            if step < min_length:
                root_logits_masked[:, eos_id] = float('-inf')

            # 3d. WINDOW REPETITION SUPPRESSION
            # (i)  Consecutive: token sama dengan step terakhir → langsung suppress
            # (ii) Window freq: token ≥2x dalam 6 step terakhir → suppress
            if len(generated_root_ids) >= 1:
                # (i) Consecutive repeat suppression
                last_ids = generated_root_ids[-1]  # (B,)
                for b in range(B):
                    tid = last_ids[b].item()
                    if tid > 3:  # jangan suppress special tokens via ini
                        root_logits_masked[b, tid] = float('-inf')

            if len(generated_root_ids) >= 2:
                window = generated_root_ids[-6:]  # max 6 step terakhir
                past_window = torch.stack(window, dim=1)  # (B, W)
                for b in range(B):
                    freq = torch.bincount(past_window[b],
                                          minlength=root_logits_masked.size(1))
                    # Token yang muncul ≥2x dalam window → hard suppress
                    suppress = (freq >= 2).nonzero(as_tuple=True)[0]
                    if suppress.numel() > 0:
                        root_logits_masked[b, suppress] = float('-inf')

            if temperature > 0:
                probs_raw = F.softmax(root_logits_masked / temperature, dim=-1)  # (B, V)

                # 3e. ENTROPY FLOOR: anti-repetition via distribusi
                entropy = -(probs_raw * (probs_raw + 1e-9).log()).sum(dim=-1)  # (B,)
                max_entropy = torch.log(torch.tensor(
                    probs_raw.size(-1), dtype=torch.float, device=device))
                entropy_ratio = entropy / (max_entropy + 1e-9)
                low_entropy_mask = (entropy_ratio < 0.05).unsqueeze(1)
                uniform = torch.ones_like(probs_raw) / probs_raw.size(-1)
                probs_mixed = torch.where(low_entropy_mask,
                                          0.8 * probs_raw + 0.2 * uniform,
                                          probs_raw)
                root_ids = torch.multinomial(probs_mixed, 1).squeeze(-1)  # (B,)
            else:
                root_ids = root_logits_masked.argmax(dim=-1)  # (B,)

            affix_ids = affix_logits.argmax(dim=-1)  # (B,)

            # 4. EOS check
            eos_logit = self.eos_predictor(state_next).squeeze(-1)  # (B,)
            eos_prob = torch.sigmoid(eos_logit)

            # 5. Update rolling context: alpha-weighted
            context_t = alpha * context_t + (1 - alpha) * state_next

            # 6. Store
            generated_root_ids.append(root_ids)
            generated_affix_ids.append(affix_ids)
            generated_states.append(state_next)
            eos_probs.append(eos_prob)

            # 7. Check if all sequences have ended (suppress EOS before min_length)
            if step >= min_length:
                active_mask = active_mask & (eos_prob < 0.5) & (root_ids != eos_id)
                if not active_mask.any():
                    break

            # 8. Advance state
            state_t = state_next

        return {
            "generated_root_ids": torch.stack(generated_root_ids, dim=1),    # (B, T)
            "generated_affix_ids": torch.stack(generated_affix_ids, dim=1),  # (B, T)
            "generated_states": torch.stack(generated_states, dim=1),        # (B, T, d)
            "eos_probs": torch.stack(eos_probs, dim=1),                      # (B, T)
        }

    def forward_autoregressive(
        self,
        target_states: torch.Tensor,
        target_root_ids: torch.Tensor,
        target_affix_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced autoregressive forward pass for training.

        At each step t, use ground truth state_t to predict state_{t+1}.
        This trains Phi_seq to learn proper state transitions.

        Args:
            target_states   : (B, L, d_total) - ground truth BSU states from encoding
            target_root_ids : (B, L) - ground truth root ids
            target_affix_ids: (B, L) - ground truth affix ids

        Returns:
            dict with ar_root_logits, ar_affix_logits, transition_states
        """
        B, L, d = target_states.shape
        alpha = self.config.alpha_context

        # prompt_anchor untuk training = mean dari semua posisi target
        # Ini konsisten dengan cara generate() menggunakan prompt mean
        prompt_anchor = target_states.mean(dim=1)  # (B, d)

        # Initialize context from first position
        context_t = target_states[:, 0, :]  # (B, d)

        ar_root_logits = []
        ar_affix_logits = []
        transition_states = []

        for t in range(L - 1):
            state_t = target_states[:, t, :]  # (B, d) - teacher forcing

            # Evolve via Phi_seq dengan prompt anchor
            state_next = self.phi_seq(state_t, context_t, prompt_anchor=prompt_anchor)  # (B, d)

            # Decode
            h_decode = self.state_to_bsu(state_next)
            root_logits = self.root_predictor(h_decode.unsqueeze(1)).squeeze(1)  # (B, V)
            affix_logits = self.affix_predictor(h_decode.unsqueeze(1)).squeeze(1)  # (B, A)

            ar_root_logits.append(root_logits)
            ar_affix_logits.append(affix_logits)
            transition_states.append(state_next)

            # Update rolling context
            context_t = alpha * context_t + (1 - alpha) * state_t

        if not ar_root_logits:
            # Edge case: L=1
            return {
                "ar_root_logits": torch.zeros(B, 0, self.config.vocab_size, device=target_states.device),
                "ar_affix_logits": torch.zeros(B, 0, self.config.affix_vocab_size, device=target_states.device),
                "transition_states": torch.zeros(B, 0, d, device=target_states.device),
            }

        return {
            "ar_root_logits": torch.stack(ar_root_logits, dim=1),      # (B, L-1, V)
            "ar_affix_logits": torch.stack(ar_affix_logits, dim=1),    # (B, L-1, A)
            "transition_states": torch.stack(transition_states, dim=1), # (B, L-1, d)
        }

    def decode(
        self,
        h: torch.Tensor,
        root_vocab_inv: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """
        Single-pass decode (non-autoregressive, for backward compat).

        Args:
            h : (B, L, d_total)
            root_vocab_inv : {id: word} for text reconstruction

        Returns:
            dict with predicted ids and optional text
        """
        root_logits = self.root_predictor(h)
        affix_logits = self.affix_predictor(h)
        role_logits = self.role_predictor(h)

        root_ids = root_logits.argmax(dim=-1)
        affix_ids = affix_logits.argmax(dim=-1)
        role_ids = role_logits.argmax(dim=-1)

        result = {
            "root_ids": root_ids,
            "affix_ids": affix_ids,
            "role_ids": role_ids,
            "root_logits": root_logits,
        }

        if root_vocab_inv:
            texts = []
            for b in range(root_ids.size(0)):
                words = []
                for t in range(root_ids.size(1)):
                    wid = root_ids[b, t].item()
                    word = root_vocab_inv.get(wid, "<UNK>")
                    if word in ("<PAD>", "<EOS>"):
                        break
                    if word not in ("<BOS>",):
                        words.append(word)
                texts.append(" ".join(words))
            result["texts"] = texts

        return result
