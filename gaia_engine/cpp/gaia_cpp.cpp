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
//
// MTP speculative decoding (GAIA_Project-ojwf): enable_mtp() loads a draft
// "assistant" (e.g. google/gemma-4-12B-it-assistant, 4 layers) that reads the
// target's hidden states and proposes the next few tokens; the target checks
// them all in one batch. Every token still comes from the target's own
// sampler, accepted only while it matches the draft, so the output follows the
// same distribution as without MTP (greedy: byte-identical on CPU). Gemma 4's
// assistant shares the target's KV cache, so prefix reuse and snapshots stay
// valid. Driven by llama.cpp's common speculative driver (b11804+).

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
#include "ggml-backend.h"
#include "llama-ext.h"   // llama_set_embeddings_nextn (staging API, used to switch MTP off)
#include "common.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "hidden_state_capture.hpp"

namespace py = pybind11;

// Free memory on the first GPU device (bytes), 0 if there is none. The
// backend's VRAM is measured as the drop in free memory across its own
// allocations (frlp): the gearbox budgets gears from it, and reported 0 made
// it count Core's whole GGUF as an outside tenant.
static size_t gaia_gpu_free_bytes() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            return free;
        }
    }
    return 0;
}
static long long gaia_gpu_delta(size_t before) {
    size_t after = gaia_gpu_free_bytes();
    return before > after ? static_cast<long long>(before - after) : 0;
}

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
    // ojwf: MTP draft tokens proposed / accepted in this request (0 = MTP off).
    int draft_tokens = 0;
    int draft_accepted = 0;
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
    //   n_threads     — CPU threads for generation (-1 = auto = hw_concurrency/2)
    //   n_threads_batch — CPU threads for reading the prompt (-1 = n_threads).
    //                   Generation is memory-bandwidth bound and peaks around
    //                   the physical core count; prompt reading is compute
    //                   bound and keeps scaling (ojwf: 12B Q8 on a 16-core
    //                   9950X3D, prefill 57 tok/s at 16 threads, 78 at 24).
    LlamaCppBackend(
        const std::string& model_path,
        int n_gpu_layers,
        std::vector<int> capture_layers,
        int n_ctx,
        int n_threads,
        int n_threads_batch,
        const std::string& kv_type
    )
        : model_path_(model_path), n_gpu_layers_(n_gpu_layers)
    {
        llama_backend_init();
        const size_t gpu_free0 = n_gpu_layers != 0 ? gaia_gpu_free_bytes() : 0;

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
        cparams.n_threads_batch = (n_threads_batch > 0) ? n_threads_batch : cparams.n_threads;

        // KV cache precision (frlp): "f16" (default) or "q8_0", which halves
        // the cache (12B at 8k context: 2.5 -> ~1.3 GB of VRAM). A quantized V
        // cache needs flash attention, so it is switched on explicitly.
        if (!kv_type.empty() && kv_type != "f16") {
            ggml_type t;
            if (kv_type == "q8_0") t = GGML_TYPE_Q8_0;
            else if (kv_type == "bf16") t = GGML_TYPE_BF16;
            else if (kv_type == "f32") t = GGML_TYPE_F32;
            else throw std::runtime_error("unsupported kv_type '" + kv_type + "' (f16, q8_0, bf16, f32)");
            cparams.type_k = t;
            cparams.type_v = t;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        kv_type_ = kv_type.empty() ? "f16" : kv_type;

        // Register the hidden state eval callback
        capture_state_.capture_layers = std::move(capture_layers);
        cparams.cb_eval = gaia_cb_eval;
        cparams.cb_eval_user_data = &capture_state_;

        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            llama_model_free(model_);
            throw std::runtime_error("Failed to create llama context");
        }
        if (n_gpu_layers != 0) vram_main_ = gaia_gpu_delta(gpu_free0);
    }

    ~LlamaCppBackend() {
        _free_mtp();
        if (mtmd_) { mtmd_free(mtmd_); mtmd_ = nullptr; }
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
        bool capture_hidden,
        float repeat_penalty = 1.0f
    ) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        repeat_penalty_ = repeat_penalty;
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
        bool capture_hidden,
        float repeat_penalty = 1.0f
    ) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        repeat_penalty_ = repeat_penalty;
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

    // ── MTP speculative decoding (ojwf) ─────────────────────────────────────

    // Load an MTP draft model and turn speculative decoding on. n_max is how
    // many tokens to draft per step (3 was best for the 12B on CPU: 1.7x).
    // Returns "" on success, else why it could not be enabled (the backend
    // then runs exactly as before).
    std::string enable_mtp(const std::string& draft_path, int n_max) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _free_mtp();
        const size_t gpu_free0 = n_gpu_layers_ != 0 ? gaia_gpu_free_bytes() : 0;

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers_;
        mtp_model_ = llama_model_load_from_file(draft_path.c_str(), mparams);
        if (!mtp_model_) return "could not load draft model " + draft_path;

        if (llama_model_n_embd_out(mtp_model_) != llama_model_n_embd_out(model_)) {
            _free_mtp();
            return "draft model does not match the target (hidden width differs)";
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = llama_n_ctx(ctx_);
        // The drafter only ever decodes a few draft tokens at a time (Gemma 4's
        // assistant shares the target KV and never reads the prompt itself), so
        // a small batch keeps its compute buffer small: 512 cost ~520 MB VRAM
        // on the GPU for nothing (frlp).
        cparams.n_batch = 64;
        cparams.n_ubatch = 64;
        cparams.n_threads = llama_n_threads(ctx_);
        cparams.n_threads_batch = llama_n_threads_batch(ctx_);
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.ctx_other = ctx_;
        cparams.n_rs_seq = 0;
        mtp_ctx_ = llama_init_from_model(mtp_model_, cparams);
        if (!mtp_ctx_) { _free_mtp(); return "could not create the MTP draft context"; }

        // The verify loop trims rejected draft tokens from the KV, which needs
        // partial sequence removal. (This probe clears the KV.)
        if (common_context_can_seq_rm(ctx_) != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
            _free_mtp();
            _invalidate_kv();
            return "target context cannot remove partial sequences; MTP needs it";
        }

        common_params_speculative sp;
        sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        sp.draft.n_max = std::max(1, n_max);
        sp.draft.ctx_tgt = ctx_;
        sp.draft.ctx_dft = mtp_ctx_;
        spec_ = common_speculative_init(sp, 1);
        if (!spec_) { _free_mtp(); _invalidate_kv(); return "speculative driver failed to start"; }

        mtp_n_max_ = sp.draft.n_max;
        mtp_path_ = draft_path;
        if (n_gpu_layers_ != 0) vram_mtp_ = gaia_gpu_delta(gpu_free0);
        _invalidate_kv();
        return "";
    }

    void disable_mtp() {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        _free_mtp();
        _invalidate_kv();
    }

    py::dict mtp_stats() {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        py::dict d;
        d["enabled"] = spec_ != nullptr;
        d["draft_model"] = mtp_path_;
        d["n_max"] = mtp_n_max_;
        d["drafted_total"] = stat_drafted_;
        d["accepted_total"] = stat_accepted_;
        d["acceptance"] = stat_drafted_ ? static_cast<double>(stat_accepted_) / stat_drafted_ : 0.0;
        return d;
    }

    // ── Images and audio (frlp) ──────────────────────────────────────────────
    //
    // load_mmproj() attaches the model's multimodal projector (the mmproj GGUF
    // exported from the same weights: convert_hf_to_gguf.py --mmproj). Media
    // then go through generate_media(): the prompt carries media_marker() where
    // each image or audio clip belongs, and media[i] holds that file's bytes
    // (jpg/png/bmp/gif; wav/mp3/flac). image_tokens > 0 pins every image to that
    // many tokens; Gemma 4 12B's HF processor uses 280 (max_soft_tokens), and
    // llama.cpp's own default gave fewer and lost a vision test.

    std::string load_mmproj(const std::string& path, int image_tokens) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        if (mtmd_) { mtmd_free(mtmd_); mtmd_ = nullptr; vram_mmproj_ = 0; }
        const size_t gpu_free0 = n_gpu_layers_ != 0 ? gaia_gpu_free_bytes() : 0;
        mtmd_context_params p = mtmd_context_params_default();
        p.use_gpu = n_gpu_layers_ != 0;
        p.print_timings = false;
        p.n_threads = llama_n_threads_batch(ctx_);
        p.warmup = false;
        if (image_tokens > 0) { p.image_min_tokens = image_tokens; p.image_max_tokens = image_tokens; }
        mtmd_ = mtmd_init_from_file(path.c_str(), model_, p);
        if (!mtmd_) return "could not load projector " + path;
        mmproj_path_ = path;
        if (n_gpu_layers_ != 0) vram_mmproj_ = gaia_gpu_delta(gpu_free0);
        return "";
    }

    std::string media_marker() const { return mtmd_ ? std::string(mtmd_get_marker(mtmd_)) : std::string(); }

    py::dict media_info() const {
        py::dict d;
        d["loaded"] = mtmd_ != nullptr;
        d["mmproj"] = mmproj_path_;
        d["vision"] = mtmd_ ? mtmd_support_vision(mtmd_) : false;
        d["audio"] = mtmd_ ? mtmd_support_audio(mtmd_) : false;
        return d;
    }

    GenerateResult generate_media(
        const std::string& prompt,
        const std::vector<std::string>& media,
        std::function<void(const std::string&)> token_cb,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        bool capture_hidden,
        float repeat_penalty = 1.0f
    ) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        repeat_penalty_ = repeat_penalty;
        if (!mtmd_) throw std::runtime_error("no projector loaded (load_mmproj)");

        std::vector<mtmd_bitmap*> bitmaps;
        auto free_bitmaps = [&]() { for (auto* b : bitmaps) mtmd_bitmap_free(b); bitmaps.clear(); };
        for (size_t i = 0; i < media.size(); ++i) {
            mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(
                mtmd_, reinterpret_cast<const unsigned char*>(media[i].data()), media[i].size(),
                false, mtmd_helper_init_opt_default());
            if (w.video_ctx) mtmd_helper_video_free(w.video_ctx);  // video isn't supported here
            if (!w.bitmap) {
                free_bitmaps();
                throw std::runtime_error("media " + std::to_string(i) + ": not a supported image or audio file");
            }
            bitmaps.push_back(w.bitmap);
        }

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        mtmd_input_text text{prompt.c_str(), prompt.size(), /*add_special=*/true, /*parse_special=*/true};
        int32_t rc = mtmd_tokenize(mtmd_, chunks, &text,
                                   const_cast<const mtmd_bitmap**>(bitmaps.data()), bitmaps.size());
        free_bitmaps();
        if (rc != 0) {
            mtmd_input_chunks_free(chunks);
            throw std::runtime_error(rc == 1 ? "media markers in the prompt don't match the media given"
                                             : "media preprocessing failed");
        }

        GenerateResult result;
        result.prompt_tokens = static_cast<int>(mtmd_helper_get_n_tokens(chunks));
        std::vector<llama_token> text_tokens;  // for the repetition penalty
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            size_t n = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(mtmd_input_chunks_get(chunks, i), &n);
            if (t) text_tokens.insert(text_tokens.end(), t, t + n);
        }
        result.evaluated_tokens = result.prompt_tokens;

        // Media aren't part of the token-based prefix cache: read fresh, and
        // leave the cache empty afterwards (snapshots of text prompts stay).
        llama_memory_clear(llama_get_memory(ctx_), true);
        kv_tokens_.clear();

        if (capture_hidden) {
            capture_state_.reset_captures();
            capture_state_.active.store(true, std::memory_order_release);
        }
        llama_pos n_past = 0;
        rc = mtmd_helper_eval_chunks(mtmd_, ctx_, chunks, 0, 0,
                                     static_cast<int32_t>(llama_n_batch(ctx_)), true, &n_past);
        if (capture_hidden) capture_state_.active.store(false, std::memory_order_release);
        mtmd_input_chunks_free(chunks);
        if (rc != 0) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            throw std::runtime_error("evaluating the media prompt failed (rc=" + std::to_string(rc) + ")");
        }
        ++stat_requests_;
        stat_evaluated_ += result.evaluated_tokens;

        llama_sampler* sampler = _build_sampler(temperature, top_p, top_k, text_tokens);
        std::string full_text;
        int n_generated = 0;
        for (int step = 0; step < max_tokens; ++step) {
            llama_token tok = llama_sampler_sample(sampler, ctx_, -1);
            if (llama_vocab_is_eog(vocab_, tok)) break;
            std::string piece = _token_to_piece(tok);
            full_text += piece;
            ++n_generated;
            if (token_cb && !piece.empty()) token_cb(piece);
            if (capture_hidden) {
                capture_state_.reset_captures();
                capture_state_.active.store(true, std::memory_order_release);
            }
            llama_batch next = llama_batch_get_one(&tok, 1);
            int ret = llama_decode(ctx_, next);
            if (capture_hidden) capture_state_.active.store(false, std::memory_order_release);
            if (ret != 0) break;
        }
        llama_sampler_free(sampler);
        if (capture_hidden) result.hidden_states = _copy_captures();

        llama_memory_clear(llama_get_memory(ctx_), true);
        kv_tokens_.clear();
        result.text = std::move(full_text);
        result.completion_tokens = n_generated;
        return result;
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
    bool has_gpu()      const { return n_gpu_layers_ != 0; }  // layers offloaded (frlp; was a TODO returning false)
    std::string model_path() const { return model_path_; }
    std::string kv_type() const { return kv_type_; }
    // MB of GPU memory this backend allocated (model + context, MTP draft,
    // projector), measured at load; 0 on the CPU.
    long long vram_mb() const { return (vram_main_ + vram_mtp_ + vram_mmproj_) / (1024 * 1024); }

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
    std::string kv_type_ = "f16";
    long long vram_main_ = 0, vram_mtp_ = 0, vram_mmproj_ = 0;  // bytes
    int n_gpu_layers_ = 0;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    // ── Media state (frlp) ────────────────────────────────────────────────
    mtmd_context* mtmd_ = nullptr;
    std::string mmproj_path_;
    float repeat_penalty_ = 1.0f;  // set per request, under inference_mtx_

    // ── MTP state (ojwf) ──────────────────────────────────────────────────
    llama_model* mtp_model_ = nullptr;
    llama_context* mtp_ctx_ = nullptr;
    common_speculative* spec_ = nullptr;
    std::string mtp_path_;
    int mtp_n_max_ = 0;
    long long stat_drafted_ = 0, stat_accepted_ = 0;

    void _free_mtp() {
        if (spec_) { common_speculative_free(spec_); spec_ = nullptr; }
        if (mtp_ctx_) { llama_free(mtp_ctx_); mtp_ctx_ = nullptr; }
        if (mtp_model_) { llama_model_free(mtp_model_); mtp_model_ = nullptr; }
        // The driver switched the target to emitting next-n embeddings; a
        // plain decode doesn't need them.
        if (ctx_) llama_set_embeddings_nextn(ctx_, false, false);
        mtp_path_.clear();
        mtp_n_max_ = 0;
        vram_mtp_ = 0;
    }

    // Copy what the hidden-state callback captured (caller decides when the
    // capture belongs to a real token of the reply).
    std::unordered_map<int, std::vector<float>> _copy_captures() {
        std::unordered_map<int, std::vector<float>> out;
        std::lock_guard<std::mutex> lk(capture_state_.mtx);
        for (auto& [layer_idx, cap] : capture_state_.captures)
            if (!cap.data.empty()) out[layer_idx] = cap.data;
        return out;
    }

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
    // in the KV; the caller evaluates the remainder. At least `n_tail` final
    // prompt tokens are always left to evaluate.
    size_t _prepare_kv(const std::vector<llama_token>& prompt, bool& restored, size_t n_tail = 1) {
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
        // first reply token, and logits aren't part of the cache. (MTP keeps
        // one more: its drafter needs the target's hidden state for the token
        // before the last, which isn't cached either.)
        if (reuse + n_tail > prompt.size()) reuse = prompt.size() - n_tail;
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
    // repeat_penalty_ != 1 (frlp): penalize every token already in the
    // context, prompt included, as HF's RepetitionPenaltyLogitsProcessor does
    // over input_ids (the GAIA engine's torch path defaults to 1.1). It runs
    // first, like HF's logits processors before its warpers.
    llama_sampler* _build_sampler(float temperature, float top_p, int top_k,
                                  const std::vector<llama_token>& history = {}) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(sparams);
        const bool penalize = repeat_penalty_ > 0.0f && repeat_penalty_ != 1.0f;
        if (penalize) {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab_), static_cast<int32_t>(llama_n_ctx(ctx_)),
                repeat_penalty_, 0.0f, 0.0f));
        }

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
        if (penalize) for (llama_token t : history) llama_sampler_accept(chain, t);
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

        // ojwf: speculative path. Not for full-prompt residual capture (SAE
        // atlas), which needs every prompt token's states from a fresh read.
        if (spec_ && prompt_tokens.size() >= 2
                && !(capture_hidden && capture_state_.all_tokens.load(std::memory_order_acquire))) {
            return _generate_mtp(prompt_tokens, token_cb, max_tokens, temperature, top_p, top_k, capture_hidden);
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
        llama_sampler* sampler = _build_sampler(temperature, top_p, top_k, prompt_tokens);
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

    // ── MTP generation loop (ojwf) ────────────────────────────────────────
    // Follows llama.cpp's examples/speculative-simple, minus checkpoints
    // (enable_mtp requires partial KV removal). The prompt is read up to its
    // last token; each step then evaluates [last token, draft...] in one
    // batch, keeps the target's sampled tokens while they match the draft,
    // and trims the rejected tail from the KV.
    GenerateResult _generate_mtp(
        const std::vector<llama_token>& prompt_tokens,
        std::function<void(const std::string&)> token_cb,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        bool capture_hidden
    ) {
        GenerateResult result;
        result.prompt_tokens = static_cast<int>(prompt_tokens.size());
        llama_memory_t mem = llama_get_memory(ctx_);
        const llama_seq_id seq = 0;

        // ── Prompt eval: everything but the last token ─────────────────────
        size_t reuse = 0;
        bool restored = false;
        if (prefix_reuse_) {
            reuse = _prepare_kv(prompt_tokens, restored, /*n_tail=*/2);
        } else {
            llama_memory_clear(mem, true);
            kv_tokens_.clear();
        }
        const size_t n_head = prompt_tokens.size() - 1;  // tokens put in the KV now

        if (capture_hidden) {
            capture_state_.reset_captures();
            capture_state_.active.store(true, std::memory_order_release);
        }
        common_batch batch_prompt(ctx_);
        for (size_t i = reuse; i < n_head; ++i)
            batch_prompt.add(prompt_tokens[i], static_cast<llama_pos>(i), seq, false);
        int ret = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());
        if (capture_hidden) capture_state_.active.store(false, std::memory_order_release);
        if (ret != 0) {
            _invalidate_kv();
            throw std::runtime_error("llama_process failed on prompt (ret=" + std::to_string(ret) + ")");
        }
        if (!common_speculative_process(spec_, batch_prompt)) {
            _invalidate_kv();
            throw std::runtime_error("MTP: speculative driver rejected the prompt batch");
        }
        // Hidden states fallback: the state at the end of the prompt read.
        if (capture_hidden) result.hidden_states = _copy_captures();

        llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.begin() + n_head);
        kv_tokens_ = prompt_tgt;
        result.reused_tokens = static_cast<int>(reuse);
        result.evaluated_tokens = static_cast<int>(n_head - reuse) + 1;  // + the last token, read in step 1
        result.snapshot_restored = restored;
        ++stat_requests_;
        stat_reused_ += static_cast<long long>(reuse);
        stat_evaluated_ += result.evaluated_tokens;
        stat_restores_ += restored;
        if (prefix_reuse_ && static_cast<size_t>(result.evaluated_tokens) >= kMinSnapshotTokens) {
            _take_snapshot(/*pinned=*/false);
        }

        common_speculative_begin(spec_, seq, prompt_tgt);

        // ── Draft / verify loop ────────────────────────────────────────────
        llama_sampler* sampler = _build_sampler(temperature, top_p, top_k, prompt_tokens);
        llama_token id_last = prompt_tokens.back();
        int n_past = static_cast<int>(n_head);
        const int n_ctx = static_cast<int>(llama_n_ctx(ctx_));
        common_batch batch_tgt(ctx_);
        llama_tokens draft;
        std::string full_text;
        int n_generated = 0;
        bool done = false;

        while (!done && n_generated < max_tokens) {
            int n_draft_max = std::min(n_ctx - n_past - 2, max_tokens - n_generated - 1);
            draft.clear();
            if (n_draft_max > 0) {
                common_speculative_get_draft_params(spec_, seq) = {
                    /* .drafting = */ true,
                    /* .n_max    = */ n_draft_max,
                    /* .pos0     = */ n_past,
                    /* .id_last  = */ id_last,
                    /* .prompt   = */ &prompt_tgt,
                    /* .result   = */ &draft,
                };
                common_speculative_draft(spec_);
            }

            batch_tgt.clear();
            batch_tgt.add(id_last, n_past, seq, true);
            for (size_t i = 0; i < draft.size(); ++i)
                batch_tgt.add(draft[i], n_past + 1 + static_cast<int>(i), seq, true);

            if (capture_hidden) {
                capture_state_.reset_captures();
                capture_state_.active.store(true, std::memory_order_release);
            }
            ret = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
            if (capture_hidden) capture_state_.active.store(false, std::memory_order_release);
            if (ret != 0 || !common_speculative_process(spec_, batch_tgt)) {
                _invalidate_kv();  // context full or error: stop, don't trust the cache
                break;
            }
            ++n_past;  // id_last is now in the KV

            // The target samples each position itself; a draft token counts
            // only if the target picked the same one.
            std::vector<llama_token> ids;
            for (size_t i = 0; i <= draft.size(); ++i) {
                llama_token tok = llama_sampler_sample(sampler, ctx_, static_cast<int32_t>(i));
                ids.push_back(tok);
                if (i < draft.size() && tok != draft[i]) break;
            }
            const size_t n_acc = ids.size() - 1;
            common_speculative_accept(spec_, seq, static_cast<uint16_t>(n_acc));
            result.draft_tokens += static_cast<int>(draft.size());
            result.draft_accepted += static_cast<int>(n_acc);
            // Every batch row was a real token: keep its last-token states.
            if (capture_hidden && n_acc == draft.size()) result.hidden_states = _copy_captures();

            prompt_tgt.push_back(id_last);  // in the KV
            for (size_t i = 0; i < ids.size(); ++i) {
                id_last = ids[i];
                if (llama_vocab_is_eog(vocab_, id_last)) { done = true; break; }
                std::string piece = _token_to_piece(id_last);
                full_text += piece;
                ++n_generated;
                if (token_cb && !piece.empty()) token_cb(piece);
                if (n_generated >= max_tokens) { done = true; break; }
                if (i + 1 < ids.size()) prompt_tgt.push_back(id_last);  // accepted draft: in the KV
            }
            // Drop rejected draft tokens (and anything past the stop point).
            n_past = static_cast<int>(prompt_tgt.size());
            llama_memory_seq_rm(mem, seq, n_past, -1);
            kv_tokens_ = prompt_tgt;
        }

        llama_sampler_free(sampler);
        stat_drafted_ += result.draft_tokens;
        stat_accepted_ += result.draft_accepted;

        if (!prefix_reuse_) {
            llama_memory_clear(mem, true);
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
        .def_readonly("draft_tokens",      &GenerateResult::draft_tokens)
        .def_readonly("draft_accepted",    &GenerateResult::draft_accepted)
        .def_property_readonly("hidden_states", [](const GenerateResult& r) {
            // Convert raw float vectors → numpy arrays (GIL is held here)
            return _hidden_states_to_numpy(r.hidden_states);
        });

    py::class_<LlamaCppBackend>(m, "LlamaCppBackend")
        .def(py::init<const std::string&, int, std::vector<int>, int, int, int, const std::string&>(),
            py::arg("model_path"),
            py::arg("n_gpu_layers")   = 0,
            py::arg("capture_layers") = std::vector<int>{},
            py::arg("n_ctx")          = 4096,
            py::arg("n_threads")      = -1,
            py::arg("n_threads_batch") = -1,
            py::arg("kv_type")        = std::string("f16"))

        .def("generate",
            [](LlamaCppBackend& self,
               const std::string& prompt,
               int max_tokens,
               float temperature,
               float top_p,
               int top_k,
               bool capture_hidden,
               float repeat_penalty) {
                // pybind11 holds the GIL when entering this lambda.
                // Release it during the C++ inference loop so other Python
                // threads can run. The GIL is re-acquired when the
                // py::gil_scoped_release destructor fires, before we
                // return the result to Python for conversion.
                GenerateResult result;
                {
                    py::gil_scoped_release release;
                    result = self.generate(prompt, max_tokens, temperature, top_p, top_k, capture_hidden,
                                           repeat_penalty);
                }
                return result;
            },
            py::arg("prompt"),
            py::arg("max_tokens")     = 512,
            py::arg("temperature")    = 0.7f,
            py::arg("top_p")          = 0.9f,
            py::arg("top_k")          = 0,
            py::arg("capture_hidden") = false,
            py::arg("repeat_penalty") = 1.0f)

        .def("generate_stream",
            [](LlamaCppBackend& self,
               const std::string& prompt,
               py::object token_callback,
               int max_tokens,
               float temperature,
               float top_p,
               int top_k,
               bool capture_hidden,
               float repeat_penalty) {
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
                        prompt, cpp_callback, max_tokens, temperature, top_p, top_k, capture_hidden,
                        repeat_penalty);
                }
                return result;
            },
            py::arg("prompt"),
            py::arg("token_callback"),
            py::arg("max_tokens")     = 512,
            py::arg("temperature")    = 0.7f,
            py::arg("top_p")          = 0.9f,
            py::arg("top_k")          = 0,
            py::arg("capture_hidden") = false,
            py::arg("repeat_penalty") = 1.0f)

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
        // ojwf: MTP speculative decoding
        .def("enable_mtp",         &LlamaCppBackend::enable_mtp,
             py::arg("draft_path"), py::arg("n_max") = 3,
             py::call_guard<py::gil_scoped_release>())
        .def("disable_mtp",        &LlamaCppBackend::disable_mtp,
             py::call_guard<py::gil_scoped_release>())
        .def("mtp_stats",          &LlamaCppBackend::mtp_stats)
        // frlp: images and audio via libmtmd
        .def("load_mmproj",        &LlamaCppBackend::load_mmproj,
             py::arg("path"), py::arg("image_tokens") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("media_marker",       &LlamaCppBackend::media_marker)
        .def("media_info",         &LlamaCppBackend::media_info)
        .def("generate_media",
            [](LlamaCppBackend& self,
               const std::string& prompt,
               std::vector<std::string> media,      // file bytes (bytes objects)
               py::object token_callback,           // None for non-streaming
               int max_tokens, float temperature, float top_p, int top_k,
               bool capture_hidden, float repeat_penalty) {
                std::function<void(const std::string&)> cb;
                if (!token_callback.is_none()) {
                    cb = [&token_callback](const std::string& delta) {
                        py::gil_scoped_acquire acquire;
                        try { token_callback(py::cast(delta)); } catch (...) {}
                    };
                }
                GenerateResult result;
                {
                    py::gil_scoped_release release;
                    result = self.generate_media(prompt, media, cb, max_tokens, temperature, top_p,
                                                 top_k, capture_hidden, repeat_penalty);
                }
                return result;
            },
            py::arg("prompt"), py::arg("media"), py::arg("token_callback") = py::none(),
            py::arg("max_tokens") = 512, py::arg("temperature") = 0.7f, py::arg("top_p") = 0.9f,
            py::arg("top_k") = 0, py::arg("capture_hidden") = false,
            py::arg("repeat_penalty") = 1.0f)

        .def("n_vocab",         &LlamaCppBackend::n_vocab)
        .def("n_embd",          &LlamaCppBackend::n_embd)
        .def("n_layer",         &LlamaCppBackend::n_layer)
        .def("has_gpu",         &LlamaCppBackend::has_gpu)
        .def("model_path",      &LlamaCppBackend::model_path)
        .def("kv_type",         &LlamaCppBackend::kv_type)
        .def("vram_mb",         &LlamaCppBackend::vram_mb)
        .def("architecture",    &LlamaCppBackend::architecture)
        .def("capture_layers",  &LlamaCppBackend::capture_layers)
        .def("set_capture_layers", &LlamaCppBackend::set_capture_layers)
        .def("capture_all_tokens", &LlamaCppBackend::capture_all_tokens)
        .def("set_capture_all_tokens", &LlamaCppBackend::set_capture_all_tokens,
            py::arg("enabled"));
}
