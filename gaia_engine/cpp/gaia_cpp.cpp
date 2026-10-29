// gaia_cpp.cpp — pybind11 extension: LlamaCppBackend
//
// In-process GGUF inference backend with hidden state extraction.
// Wraps llama.cpp directly — no subprocess, no HTTP round-trips.
//
// Replaces the llama-server subprocess path in manager.py for GGUF models,
// providing the same generation speed (~97 tok/s CPU) plus full hidden state
// access via cb_eval tensor callbacks (SAE/polygraph compatible).
//
// Architecture:
//   Python (GAIAEngine / manager.py)
//     └── GaiaCppBackendAdapter (backend.py) — Python wrapper
//         └── LlamaCppBackend (this file) — C++ pybind11 class
//             └── llama.cpp libllama.so — inference engine
//                 └── cb_eval callback → HiddenStateCaptureState
//
// Thread safety: ONE active generate/generate_stream at a time per instance.
// manager.py's _inference_semaphore provides the outer guard.
// The C++ inference_mtx_ catches any internal races.
//
// KV reuse (GAIA_Project-d12t): the KV cache is kept between requests.
// Reading the prompt is the dominant cost on CPU (~155 tok/s vs ~10 tok/s
// generation, but GAIA prompts are 1.5-3k tokens), and every request used to
// clear the cache and read its whole prompt again. Now:
//   - prefix reuse: only the tokens after the longest common prefix with
//     what is already in the KV are evaluated;
//   - snapshots: after an expensive prompt read, the KV state is copied
//     into a small in-memory set (LRU; a loaded baseline is pinned). A
//     request that shares more with a snapshot than with the live cache
//     restores it first, so callers that alternate between different
//     prompts don't evict each other's prefix;
//   - save_state/load_state persist the live KV to disk, so a baseline
//     (e.g. a system prompt) survives a restart.
// The cache is dropped whenever it could be stale: LoRA changes, decode
// errors, and requests that capture every prompt token's hidden state.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "llama.h"
#include "hidden_state_capture.hpp"

namespace py = pybind11;

// ── GenerateResult ────────────────────────────────────────────────────────────

// Internal result uses raw C++ vectors — no Python objects.
// Converted to numpy in the pybind11 wrappers AFTER re-acquiring the GIL.
struct GenerateResult {
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    // d12t: how the prompt was read — tokens taken from the KV cache vs
    // evaluated now, and whether a snapshot was restored to get them.
    int reused_tokens = 0;
    int evaluated_tokens = 0;
    bool snapshot_restored = false;
    // {layer_idx → raw float vector}, converted to numpy after GIL re-acquisition
    std::unordered_map<int, std::vector<float>> hidden_states;
};

// ── LlamaCppBackend ───────────────────────────────────────────────────────────

class LlamaCppBackend {
public:
    // Args:
    //   model_path    — path to .gguf file
    //   n_gpu_layers  — layers to offload to GPU (0=CPU, -1=all)
    //   capture_layers— layer indices for hidden state extraction (empty=none)
    //   n_ctx         — context window size
    //   n_threads     — CPU threads for inference (-1 = auto = hw_concurrency/2)
    LlamaCppBackend(
        const std::string& model_path,
        int n_gpu_layers,
        std::vector<int> capture_layers,
        int n_ctx,
        int n_threads
    )
        : model_path_(model_path)
    {
        llama_backend_init();

        // ── Model params ────────────────────────────────────────────────────
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;

        model_ = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model_) {
            throw std::runtime_error("Failed to load GGUF model: " + model_path);
        }

        vocab_ = llama_model_get_vocab(model_);

        // ── Context params ─────────────────────────────────────────────────
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(n_ctx);
        cparams.n_batch = static_cast<uint32_t>(n_ctx);
        cparams.n_threads = (n_threads > 0)
            ? n_threads
            : static_cast<int>(std::thread::hardware_concurrency() / 2);
        cparams.n_threads_batch = cparams.n_threads;

        // Register the hidden state eval callback
        capture_state_.capture_layers = std::move(capture_layers);
        cparams.cb_eval = gaia_cb_eval;
        cparams.cb_eval_user_data = &capture_state_;

        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            llama_model_free(model_);
            throw std::runtime_error("Failed to create llama context");
        }
    }

    ~LlamaCppBackend() {
        lora_adapters_.clear();  // adapter memory managed by llama_model_free in b8250+
        if (ctx_)   { llama_free(ctx_); ctx_ = nullptr; }
        if (model_) { llama_model_free(model_); model_ = nullptr; }
        llama_backend_free();
    }

    // ── generate() ───────────────────────────────────────────────────────────
    // Non-streaming. Returns full text + optional hidden states.
    GenerateResult generate(
        const std::string& prompt,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        bool capture_hidden
    ) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        return _generate_impl(prompt, nullptr, max_tokens, temperature, top_p, top_k, capture_hidden);
    }

    // ── generate_stream() ────────────────────────────────────────────────────
    // Streaming. Calls token_callback(delta_text) for each token from C++.
    // To avoid GIL deadlock, use py::gil_scoped_release before calling this
    // from Python, and ensure token_callback re-acquires the GIL internally.
    //
    // Returns GenerateResult after generation completes (same as generate()).
    GenerateResult generate_stream(
        const std::string& prompt,
        std::function<void(const std::string&)> token_callback,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        bool capture_hidden
    ) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        return _generate_impl(prompt, token_callback, max_tokens, temperature, top_p, top_k, capture_hidden);
    }

    // ── LoRA adapter management ───────────────────────────────────────────────
    //
    // llama.cpp b8250 API:
    //   llama_adapter_lora_init(model, path) → adapter*  (parse GGUF LoRA file)
    //   llama_set_adapter_lora(ctx, adapter, scale)      (activate on context)
    //   llama_rm_adapter_lora(ctx, adapter)              (deactivate from context)
    //   llama_clear_adapter_lora(ctx)                    (remove all from context)
    //   llama_adapter_lora_free(adapter)                 (free memory)
    //
    // Multiple adapters can be active simultaneously with different scales.
    // The adapter modifies the model weights in-place during forward pass.

    bool load_lora(const std::string& path, float scale) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _invalidate_kv();  // cached KV was computed with the old adapters

        llama_adapter_lora* adapter = llama_adapter_lora_init(model_, path.c_str());
        if (!adapter) {
            return false;
        }

        lora_adapters_.push_back(adapter);
        lora_scales_.push_back(scale);

        // Apply all adapters at once (b8250 plural API)
        int32_t ret = llama_set_adapters_lora(
            ctx_,
            lora_adapters_.data(),
            lora_adapters_.size(),
            lora_scales_.data()
        );
        if (ret != 0) {
            // Roll back: remove the adapter we just added
            lora_adapters_.pop_back();
            lora_scales_.pop_back();
            // Re-apply previous set
            if (!lora_adapters_.empty()) {
                llama_set_adapters_lora(ctx_, lora_adapters_.data(),
                    lora_adapters_.size(), lora_scales_.data());
            }
            return false;
        }

        return true;
    }

    void clear_lora() {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _invalidate_kv();

        // Detach all adapters from context
        llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
        lora_adapters_.clear();
        lora_scales_.clear();
        // Adapters freed automatically with model (b8250+)
    }

    // ── KV cache control (d12t) ─────────────────────────────────────────────

    void set_prefix_reuse(bool enabled) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        prefix_reuse_ = enabled;
        if (!enabled) _invalidate_kv();
    }

    // Number of unpinned in-memory snapshots kept (0 disables them). A
    // snapshot holds the full KV for its prompt: ~144 KB/token for an 8B
    // model with 36 layers and 8 KV heads, so ~430 MB at 3k tokens.
    void set_snapshot_slots(int n) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        snapshot_slots_ = std::max(0, n);
        _evict_snapshots();
    }

    void clear_kv() {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _invalidate_kv();
    }

    // Persist the live KV (the last request's prompt + reply) to `path`.
    // Returns the number of tokens saved (0 = nothing to save / failed).
    int save_state(const std::string& path) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        if (kv_tokens_.empty()) return 0;
        size_t n = llama_state_seq_save_file(ctx_, path.c_str(), 0,
                                             kv_tokens_.data(), kv_tokens_.size());
        return n > 0 ? static_cast<int>(kv_tokens_.size()) : 0;
    }

    // Load a state saved by save_state() into the live KV, and keep it as a
    // pinned snapshot (never evicted) so later requests can build on it.
    // Returns the number of tokens loaded (0 = failed; the cache is then empty).
    int load_state(const std::string& path) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _invalidate_kv();
        std::vector<llama_token> toks(static_cast<size_t>(llama_n_ctx(ctx_)));
        size_t n_tok = 0;
        size_t n = llama_state_seq_load_file(ctx_, path.c_str(), 0,
                                             toks.data(), toks.size(), &n_tok);
        if (n == 0 || n_tok == 0) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            return 0;
        }
        toks.resize(n_tok);
        kv_tokens_ = toks;
        _take_snapshot(/*pinned=*/true);
        return static_cast<int>(n_tok);
    }

    py::dict kv_stats() {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        py::dict d;
        d["prefix_reuse"] = prefix_reuse_;
        d["cached_tokens"] = static_cast<int>(kv_tokens_.size());
        d["snapshot_slots"] = snapshot_slots_;
        py::list snaps;
        size_t bytes = 0;
        for (const auto& sn : snapshots_) {
            py::dict e;
            e["tokens"] = static_cast<int>(sn.tokens.size());
            e["bytes"] = static_cast<long long>(sn.data.size());
            e["pinned"] = sn.pinned;
            snaps.append(e);
            bytes += sn.data.size();
        }
        d["snapshots"] = snaps;
        d["snapshot_bytes"] = static_cast<long long>(bytes);
        d["requests"] = stat_requests_;
        d["reused_tokens_total"] = stat_reused_;
        d["evaluated_tokens_total"] = stat_evaluated_;
        d["snapshot_restores"] = stat_restores_;
        return d;
    }

    // List active adapter count (for health/status reporting)
    int active_adapter_count() const {
        return static_cast<int>(lora_adapters_.size());
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    int n_vocab()       const { return llama_vocab_n_tokens(vocab_); }
    int n_embd()        const { return llama_model_n_embd(model_); }
    int n_layer()       const { return llama_model_n_layer(model_); }
    bool has_gpu()      const { return false; } // TODO Phase 5: check via llama_model GPU offload state
    std::string model_path() const { return model_path_; }

    // GGUF general.architecture ("gemma4", "qwen3vl", ...): the prompt format
    // follows the model, not its file name (a symlink like core.gguf says
    // nothing about what it points to).
    std::string architecture() const {
        char buf[128] = {0};
        int n = llama_model_meta_val_str(model_, "general.architecture", buf, sizeof(buf));
        return n > 0 ? std::string(buf) : std::string();
    }

    std::vector<int> capture_layers() const {
        return capture_state_.capture_layers;
    }

    void set_capture_layers(std::vector<int> layers) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        capture_state_.capture_layers = std::move(layers);
    }

    // xzi: opt into capturing ALL prompt-token residuals (n_embd × n_tokens)
    // instead of just the last token — for SAE atlas sample efficiency. Default
    // off keeps the polygraph's last-token behavior.
    void set_capture_all_tokens(bool enabled) {
        capture_state_.all_tokens.store(enabled, std::memory_order_release);
    }
    bool capture_all_tokens() const {
        return capture_state_.all_tokens.load(std::memory_order_acquire);
    }

private:
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    HiddenStateCaptureState capture_state_;
    std::mutex inference_mtx_;
    std::vector<llama_adapter_lora*> lora_adapters_;
    std::vector<float> lora_scales_;

    // ── KV reuse state (d12t) ─────────────────────────────────────────────
    struct KvSnapshot {
        std::vector<llama_token> tokens;  // what the saved KV holds
        std::vector<uint8_t> data;        // llama_state_seq_get_data blob
        bool pinned = false;              // loaded baseline: never evicted
        uint64_t last_used = 0;
    };
    // Snapshot only prompt reads at least this long: shorter ones are cheap
    // to redo (~0.4 s on CPU) and not worth the memory.
    static constexpr size_t kMinSnapshotTokens = 256;
    // A snapshot replaces the live cache only if it saves this many more
    // tokens of reading than the live cache does (restoring costs a copy).
    static constexpr size_t kMinRestoreGain = 64;

    bool prefix_reuse_ = true;
    int snapshot_slots_ = 2;
    std::vector<llama_token> kv_tokens_;  // tokens currently in seq 0's KV
    std::vector<KvSnapshot> snapshots_;
    uint64_t use_clock_ = 0;
    long long stat_requests_ = 0, stat_reused_ = 0, stat_evaluated_ = 0, stat_restores_ = 0;

    static size_t _common_prefix(const std::vector<llama_token>& a, const std::vector<llama_token>& b) {
        size_t n = std::min(a.size(), b.size()), i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }

    void _invalidate_kv() {
        llama_memory_clear(llama_get_memory(ctx_), true);
        kv_tokens_.clear();
        snapshots_.clear();
    }

    // Copy the live KV into a snapshot (replacing one for the same tokens).
    void _take_snapshot(bool pinned) {
        if (kv_tokens_.empty() || (!pinned && snapshot_slots_ <= 0)) return;
        size_t size = llama_state_seq_get_size(ctx_, 0);
        if (size == 0) return;
        KvSnapshot sn;
        sn.tokens = kv_tokens_;
        sn.data.resize(size);
        if (llama_state_seq_get_data(ctx_, sn.data.data(), size, 0) == 0) return;
        sn.pinned = pinned;
        sn.last_used = ++use_clock_;
        snapshots_.erase(std::remove_if(snapshots_.begin(), snapshots_.end(),
            [&](const KvSnapshot& o) { return o.tokens == sn.tokens; }), snapshots_.end());
        snapshots_.push_back(std::move(sn));
        _evict_snapshots();
    }

    // Keep at most snapshot_slots_ unpinned snapshots, least recently used out.
    void _evict_snapshots() {
        for (;;) {
            int unpinned = 0;
            for (const auto& s : snapshots_) unpinned += !s.pinned;
            if (unpinned <= snapshot_slots_) return;
            auto victim = snapshots_.end();
            for (auto it = snapshots_.begin(); it != snapshots_.end(); ++it)
                if (!it->pinned && (victim == snapshots_.end() || it->last_used < victim->last_used))
                    victim = it;
            snapshots_.erase(victim);
        }
    }

    // Prepare seq 0's KV for `prompt`: keep (or restore) the longest usable
    // prefix and drop the rest. Returns how many prompt tokens are already
    // in the KV; the caller evaluates the remainder.
    size_t _prepare_kv(const std::vector<llama_token>& prompt, bool& restored) {
        restored = false;
        llama_memory_t mem = llama_get_memory(ctx_);
        size_t reuse = _common_prefix(kv_tokens_, prompt);

        // A snapshot that shares clearly more of this prompt beats the live cache.
        KvSnapshot* best = nullptr;
        size_t best_n = reuse;
        for (auto& sn : snapshots_) {
            size_t n = _common_prefix(sn.tokens, prompt);
            if (n > best_n + kMinRestoreGain) { best = &sn; best_n = n; }
        }
        if (best) {
            llama_memory_clear(mem, true);
            if (llama_state_seq_set_data(ctx_, best->data.data(), best->data.size(), 0) > 0) {
                kv_tokens_ = best->tokens;
                best->last_used = ++use_clock_;
                reuse = best_n;
                restored = true;
            } else {
                kv_tokens_.clear();  // failed restore left nothing usable
                reuse = 0;
            }
        }

        // The last prompt token is always evaluated: its logits pick the
        // first reply token, and logits aren't part of the cache.
        if (reuse >= prompt.size()) reuse = prompt.size() - 1;
        if (reuse < kv_tokens_.size() && !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(reuse), -1)) {
            llama_memory_clear(mem, true);  // partial removal unsupported: start over
            reuse = 0;
        }
        kv_tokens_.resize(reuse);
        return reuse;
    }

    // ── Tokenize ──────────────────────────────────────────────────────────────
    std::vector<llama_token> _tokenize(const std::string& text, bool add_bos) {
        // First call with n_max=0 to get length
        int n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_bos, true);
        if (n < 0) n = -n;  // negative means buffer too small — take abs
        std::vector<llama_token> tokens(static_cast<size_t>(n));
        llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                       tokens.data(), n, add_bos, true);
        return tokens;
    }

    // ── Token to text ─────────────────────────────────────────────────────────
    std::string _token_to_piece(llama_token tok) {
        char buf[256];
        int n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, true);
        if (n <= 0) return "";
        return std::string(buf, static_cast<size_t>(n));
    }

    // ── Build sampler ─────────────────────────────────────────────────────────
    llama_sampler* _build_sampler(float temperature, float top_p, int top_k) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(sparams);

        if (top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        }
        if (top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        }
        if (temperature <= 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
        return chain;
    }

    // ── Core generation loop ──────────────────────────────────────────────────
    GenerateResult _generate_impl(
        const std::string& prompt,
        std::function<void(const std::string&)> token_cb,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        bool capture_hidden
    ) {
        GenerateResult result;

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens = _tokenize(prompt, true);
        result.prompt_tokens = static_cast<int>(prompt_tokens.size());

        if (prompt_tokens.empty()) {
            return result;
        }

        // ── Prompt eval ────────────────────────────────────────────────────
        // d12t: start from the cached prefix. Capturing every prompt token's
        // residuals (SAE atlas) needs them all evaluated, so it reads fresh.
        size_t reuse = 0;
        bool restored = false;
        bool full_capture = capture_hidden && capture_state_.all_tokens.load(std::memory_order_acquire);
        if (prefix_reuse_ && !full_capture) {
            reuse = _prepare_kv(prompt_tokens, restored);
        } else {
            llama_memory_clear(llama_get_memory(ctx_), true);
            kv_tokens_.clear();
        }

        if (capture_hidden) {
            capture_state_.reset_captures();
            capture_state_.active.store(true, std::memory_order_release);
        }

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data() + reuse,
            static_cast<int32_t>(prompt_tokens.size() - reuse)
        );
        int ret = llama_decode(ctx_, batch);

        if (capture_hidden) {
            capture_state_.active.store(false, std::memory_order_release);
        }

        if (ret != 0) {
            _invalidate_kv();
            throw std::runtime_error("llama_decode failed on prompt (ret=" + std::to_string(ret) + ")");
        }
        kv_tokens_ = prompt_tokens;
        result.reused_tokens = static_cast<int>(reuse);
        result.evaluated_tokens = static_cast<int>(prompt_tokens.size() - reuse);
        result.snapshot_restored = restored;
        ++stat_requests_;
        stat_reused_ += static_cast<long long>(reuse);
        stat_evaluated_ += result.evaluated_tokens;
        stat_restores_ += restored;
        // Keep a baseline of an expensive read so a later prompt sharing it
        // doesn't pay for it again after another caller has moved the cache.
        if (prefix_reuse_ && !full_capture
                && static_cast<size_t>(result.evaluated_tokens) >= kMinSnapshotTokens) {
            _take_snapshot(/*pinned=*/false);
        }

        // ── Autoregressive generation loop ────────────────────────────────
        llama_sampler* sampler = _build_sampler(temperature, top_p, top_k);
        std::string full_text;
        int n_generated = 0;

        for (int step = 0; step < max_tokens; ++step) {
            llama_token tok = llama_sampler_sample(sampler, ctx_, -1);

            if (llama_vocab_is_eog(vocab_, tok)) {
                break;
            }

            std::string piece = _token_to_piece(tok);
            full_text += piece;
            ++n_generated;

            // Fire streaming callback if provided
            if (token_cb && !piece.empty()) {
                token_cb(piece);
            }

            // Capture hidden states for the current decode step
            // We capture every token's states (last-token position in cb_eval)
            // and overwrite — caller gets the LAST generated token's states.
            if (capture_hidden) {
                capture_state_.reset_captures();
                capture_state_.active.store(true, std::memory_order_release);
            }

            llama_batch next_batch = llama_batch_get_one(&tok, 1);
            ret = llama_decode(ctx_, next_batch);

            if (capture_hidden) {
                capture_state_.active.store(false, std::memory_order_release);
            }

            if (ret != 0) {
                // Context full or error: stop gracefully, and don't trust the
                // cache for the next request.
                _invalidate_kv();
                break;
            }
            kv_tokens_.push_back(tok);  // now in the KV: a follow-up can reuse it
        }

        llama_sampler_free(sampler);

        // ── Extract captured hidden states → raw vectors ─────────────────
        // Stored as C++ vectors here; converted to numpy in pybind11 wrappers
        // AFTER the GIL is re-acquired (py::array_t requires the GIL).
        if (capture_hidden) {
            std::lock_guard<std::mutex> lk(capture_state_.mtx);
            for (auto& [layer_idx, cap] : capture_state_.captures) {
                if (cap.data.empty()) continue;
                result.hidden_states[layer_idx] = std::move(cap.data);
            }
        }

        // d12t: the KV is kept (prompt + reply) for the next request's prefix.
        if (!prefix_reuse_ || full_capture) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            kv_tokens_.clear();
        }

        result.text = std::move(full_text);
        result.completion_tokens = n_generated;
        return result;
    }
};

// ── pybind11 module ───────────────────────────────────────────────────────────

// Convert raw C++ hidden states to Python dict of numpy arrays.
// MUST be called with the GIL held (which pybind11 guarantees in def lambdas).
static py::dict _hidden_states_to_numpy(
    const std::unordered_map<int, std::vector<float>>& hidden_states
) {
    py::dict result;
    for (const auto& [layer_idx, vec] : hidden_states) {
        if (vec.empty()) continue;
        auto arr = py::array_t<float>(static_cast<ssize_t>(vec.size()));
        auto buf = arr.request();
        std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(float));
        result[py::int_(layer_idx)] = std::move(arr);
    }
    return result;
}

PYBIND11_MODULE(gaia_cpp, m) {
    m.doc() = "gaia_cpp — in-process llama.cpp backend for GAIA Engine";

    py::class_<GenerateResult>(m, "GenerateResult")
        .def_readonly("text",              &GenerateResult::text)
        .def_readonly("prompt_tokens",     &GenerateResult::prompt_tokens)
        .def_readonly("completion_tokens", &GenerateResult::completion_tokens)
        .def_readonly("reused_tokens",     &GenerateResult::reused_tokens)
        .def_readonly("evaluated_tokens",  &GenerateResult::evaluated_tokens)
        .def_readonly("snapshot_restored", &GenerateResult::snapshot_restored)
        .def_property_readonly("hidden_states", [](const GenerateResult& r) {
            // Convert raw float vectors → numpy arrays (GIL is held here)
            return _hidden_states_to_numpy(r.hidden_states);
        });

    py::class_<LlamaCppBackend>(m, "LlamaCppBackend")
        .def(py::init<const std::string&, int, std::vector<int>, int, int>(),
            py::arg("model_path"),
            py::arg("n_gpu_layers")   = 0,
            py::arg("capture_layers") = std::vector<int>{},
            py::arg("n_ctx")          = 4096,
            py::arg("n_threads")      = -1)

        .def("generate",
            [](LlamaCppBackend& self,
               const std::string& prompt,
               int max_tokens,
               float temperature,
               float top_p,
               int top_k,
               bool capture_hidden) {
                // pybind11 holds the GIL when entering this lambda.
                // Release it during the C++ inference loop so other Python
                // threads can run. The GIL is re-acquired when the
                // py::gil_scoped_release destructor fires, before we
                // return the result to Python for conversion.
                GenerateResult result;
                {
                    py::gil_scoped_release release;
                    result = self.generate(prompt, max_tokens, temperature, top_p, top_k, capture_hidden);
                }
                return result;
            },
            py::arg("prompt"),
            py::arg("max_tokens")     = 512,
            py::arg("temperature")    = 0.7f,
            py::arg("top_p")          = 0.9f,
            py::arg("top_k")          = 0,
            py::arg("capture_hidden") = false)

        .def("generate_stream",
            [](LlamaCppBackend& self,
               const std::string& prompt,
               py::object token_callback,
               int max_tokens,
               float temperature,
               float top_p,
               int top_k,
               bool capture_hidden) {
                // The token callback fires from C++ while the GIL is released.
                // It must re-acquire the GIL to call back into Python.
                // py::gil_scoped_acquire is the pybind11-native way to do this.
                auto cpp_callback = [&token_callback](const std::string& delta) {
                    py::gil_scoped_acquire acquire;
                    try {
                        token_callback(py::cast(delta));
                    } catch (...) {}
                };
                GenerateResult result;
                {
                    py::gil_scoped_release release;
                    result = self.generate_stream(
                        prompt, cpp_callback, max_tokens, temperature, top_p, top_k, capture_hidden);
                }
                return result;
            },
            py::arg("prompt"),
            py::arg("token_callback"),
            py::arg("max_tokens")     = 512,
            py::arg("temperature")    = 0.7f,
            py::arg("top_p")          = 0.9f,
            py::arg("top_k")          = 0,
            py::arg("capture_hidden") = false)

        .def("load_lora",  &LlamaCppBackend::load_lora,
            py::arg("path"), py::arg("scale") = 1.0f)
        .def("clear_lora", &LlamaCppBackend::clear_lora)
        .def("active_adapter_count", &LlamaCppBackend::active_adapter_count)
        // d12t: KV cache control
        .def("set_prefix_reuse",   &LlamaCppBackend::set_prefix_reuse, py::arg("enabled"))
        .def("set_snapshot_slots", &LlamaCppBackend::set_snapshot_slots, py::arg("n"))
        .def("clear_kv",           &LlamaCppBackend::clear_kv)
        .def("save_state",         &LlamaCppBackend::save_state, py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_state",         &LlamaCppBackend::load_state, py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("kv_stats",           &LlamaCppBackend::kv_stats)

        .def("n_vocab",         &LlamaCppBackend::n_vocab)
        .def("n_embd",          &LlamaCppBackend::n_embd)
        .def("n_layer",         &LlamaCppBackend::n_layer)
        .def("has_gpu",         &LlamaCppBackend::has_gpu)
        .def("model_path",      &LlamaCppBackend::model_path)
        .def("architecture",    &LlamaCppBackend::architecture)
        .def("capture_layers",  &LlamaCppBackend::capture_layers)
        .def("set_capture_layers", &LlamaCppBackend::set_capture_layers)
        .def("capture_all_tokens", &LlamaCppBackend::capture_all_tokens)
        .def("set_capture_all_tokens", &LlamaCppBackend::set_capture_all_tokens,
            py::arg("enabled"));
}
