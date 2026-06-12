#pragma once
// hidden_state_capture.hpp
// Header-only: cb_eval callback + hidden state capture state for gaia_cpp.
//
// The ggml_backend_sched_eval_callback fires for EVERY tensor computed during
// llama_decode(). We filter by tensor name to capture per-layer block outputs.
//
// Tensor naming in llama.cpp b8250 for Qwen2/Qwen3/Qwen3MoE:
//   "l_out-N"   — post-FFN residual (b8250+ new format, e.g. "l_out-0")
//   "blk.N.l_out" — same tensor, older llama.cpp format (kept for compat)
//   Both correspond to hidden_states[N+1] in transformers library.
//
// The callback uses ask/answer protocol:
//   ask=true  → "do you want this tensor's data?" return true to allow, false to skip
//   ask=false → tensor data is ready, no meaningful return value
//
// Thread safety: capture_state is owned by one LlamaCppBackend instance.
// The inference_mtx_ in LlamaCppBackend ensures single-call-at-a-time.
// The active flag gates captures — only set during llama_decode() calls.

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

struct LayerCapture {
    int layer_idx = -1;
    int64_t n_embd = 0;
    std::vector<float> data;  // F32 copy of the last-token hidden state
};

struct HiddenStateCaptureState {
    // Layers to capture. Empty = capture none (no overhead in cb_eval).
    std::vector<int> capture_layers;

    // Set to true immediately before llama_decode(), false immediately after.
    // The callback is a no-op when active=false.
    std::atomic<bool> active{false};

    // When true, capture ALL token positions (n_embd × n_tokens, token-major)
    // instead of just the last. Opt-in for SAE atlas recording (xzi); default
    // false preserves the last-token behavior the polygraph expects.
    std::atomic<bool> all_tokens{false};

    // Populated during llama_decode(). Read after decode returns.
    // Protected by mtx during writes from the callback.
    std::unordered_map<int, LayerCapture> captures;
    std::mutex mtx;

    void reset_captures() {
        std::lock_guard<std::mutex> lk(mtx);
        captures.clear();
    }
};

// ── Inline callback implementation ───────────────────────────────────────────
//
// Registered as llama_context_params.cb_eval with cb_eval_user_data pointing
// to a HiddenStateCaptureState*.
//
// Called by ggml scheduler for every tensor op in the computation graph.
// Must be fast — this runs on the critical path during every forward pass.

// Parse tensor name to extract layer index. Returns -1 if not a block output tensor.
// Supports both "l_out-N" (b8250+) and "blk.N.l_out" (older) formats.
static inline int _parse_layer_from_name(const char* name) {
    const char* p;
    int layer = 0;
    bool has_digit = false;

    if (name[0] == 'l' && name[1] == '_' && name[2] == 'o' &&
        name[3] == 'u' && name[4] == 't' && name[5] == '-') {
        // "l_out-N"
        p = name + 6;
        while (*p >= '0' && *p <= '9') {
            layer = layer * 10 + (*p - '0');
            ++p;
            has_digit = true;
        }
        if (!has_digit || *p != '\0') return -1;
        return layer;
    }

    if (name[0] == 'b' && name[1] == 'l' && name[2] == 'k' && name[3] == '.') {
        // "blk.N.l_out"
        p = name + 4;
        while (*p >= '0' && *p <= '9') {
            layer = layer * 10 + (*p - '0');
            ++p;
            has_digit = true;
        }
        if (!has_digit || *p != '.') return -1;
        ++p;
        if (strcmp(p, "l_out") != 0) return -1;
        return layer;
    }

    return -1;
}

static bool gaia_cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* state = reinterpret_cast<HiddenStateCaptureState*>(user_data);

    // Fast exit: if capture is not active, don't request any tensor data.
    // Returning false for ask=true means "I don't need this tensor's data synced."
    // The computation still happens — we just skip the host-memory sync.
    if (!state->active.load(std::memory_order_relaxed)) return false;
    if (state->capture_layers.empty()) return false;

    // Parse tensor name — reject non-block-output tensors immediately
    int layer = _parse_layer_from_name(t->name);
    if (layer < 0) return false;

    // Check if this layer is in the capture set
    bool should_capture = false;
    for (int cl : state->capture_layers) {
        if (cl == layer) {
            should_capture = true;
            break;
        }
    }
    if (!should_capture) return false;

    // ask=true: scheduler asking if we want this tensor's data synced to host.
    // Return true only for tensors we actually want to capture.
    if (ask) return true;

    // Tensor shape in ggml (column-major): [n_embd, n_tokens, 1, 1]
    //   ne[0] = n_embd (embedding dimension)
    //   ne[1] = n_tokens (number of tokens in batch)
    //   nb[1] = byte stride between tokens (= n_embd * element_size for packed)
    int64_t n_embd = t->ne[0];
    int64_t n_tokens = t->ne[1];
    if (n_embd <= 0 || n_tokens <= 0) return false;

    // Last token by default; ALL positions when opted in (SAE atlas, xzi). Data
    // is token-major / n_embd-contiguous: [tok0 n_embd][tok1 n_embd]... so the
    // Python side reshapes flat → (n_captured_tokens, n_embd).
    const bool grab_all = state->all_tokens.load(std::memory_order_acquire);
    const int64_t first_tok = grab_all ? 0 : (n_tokens - 1);
    const int64_t count_tok = grab_all ? n_tokens : 1;

    LayerCapture cap;
    cap.layer_idx = layer;
    cap.n_embd = n_embd;
    cap.data.resize(static_cast<size_t>(n_embd) * static_cast<size_t>(count_tok));

    if (t->type == GGML_TYPE_F32) {
        for (int64_t k = 0; k < count_tok; ++k) {
            size_t byte_offset = static_cast<size_t>(first_tok + k) * t->nb[1];
            ggml_backend_tensor_get(
                t, cap.data.data() + static_cast<size_t>(k) * static_cast<size_t>(n_embd),
                byte_offset, static_cast<size_t>(n_embd) * sizeof(float));
        }
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> f16_buf(static_cast<size_t>(n_embd));
        for (int64_t k = 0; k < count_tok; ++k) {
            size_t byte_offset = static_cast<size_t>(first_tok + k) * t->nb[1];
            ggml_backend_tensor_get(t, f16_buf.data(), byte_offset,
                                    static_cast<size_t>(n_embd) * sizeof(uint16_t));
            float* out = cap.data.data() + static_cast<size_t>(k) * static_cast<size_t>(n_embd);
            for (int64_t i = 0; i < n_embd; ++i) {
                uint16_t h = f16_buf[static_cast<size_t>(i)];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f32_bits;
                if (exp == 0) {
                    if (mant == 0) {
                        f32_bits = sign << 31;
                    } else {
                        // Denormal
                        exp = 1;
                        while (!(mant & 0x400)) { mant <<= 1; --exp; }
                        mant &= 0x3FF;
                        f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    // Inf or NaN
                    f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
                } else {
                    f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&out[static_cast<size_t>(i)], &f32_bits, sizeof(float));
            }
        }
    } else {
        // Unsupported type (e.g., Q8_0 activations — shouldn't happen for l_out)
        // Dequant is complex; skip and log
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(state->mtx);
        state->captures[layer] = std::move(cap);
    }
    return true;
}
