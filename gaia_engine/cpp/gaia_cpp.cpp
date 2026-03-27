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

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <atomic>
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

struct GenerateResult {
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    // {layer_idx → numpy array shape (n_embd,), dtype=float32}
    std::unordered_map<int, py::array_t<float>> hidden_states;
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
        cparams.n_batch = 512;
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

    bool load_lora(const std::string& path, float scale) {
        // TODO Phase 4: implement using llama_set_adapters_lora (b8250+ API)
        // The b8250 API changed from llama_set_adapter_lora (singular) to
        // llama_set_adapters_lora (plural). Stubbed until Phase 4.
        (void)path; (void)scale;
        return false;
    }

    void clear_lora() {
        // TODO Phase 4: implement using llama_set_adapters_lora with empty list
        lora_adapters_.clear();
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    int n_vocab()       const { return llama_vocab_n_tokens(vocab_); }
    int n_embd()        const { return llama_model_n_embd(model_); }
    int n_layer()       const { return llama_model_n_layer(model_); }
    bool has_gpu()      const { return false; } // TODO Phase 5: check via llama_model GPU offload state
    std::string model_path() const { return model_path_; }

    std::vector<int> capture_layers() const {
        return capture_state_.capture_layers;
    }

    void set_capture_layers(std::vector<int> layers) {
        std::lock_guard<std::mutex> lock(inference_mtx_);
        capture_state_.capture_layers = std::move(layers);
    }

private:
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    HiddenStateCaptureState capture_state_;
    std::mutex inference_mtx_;
    std::vector<llama_adapter_lora*> lora_adapters_;

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
        if (capture_hidden) {
            capture_state_.reset_captures();
            capture_state_.active.store(true, std::memory_order_release);
        }

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size())
        );
        int ret = llama_decode(ctx_, batch);

        if (capture_hidden) {
            capture_state_.active.store(false, std::memory_order_release);
        }

        if (ret != 0) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            throw std::runtime_error("llama_decode failed on prompt (ret=" + std::to_string(ret) + ")");
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
                break;  // context full or error — stop gracefully
            }
        }

        llama_sampler_free(sampler);

        // ── Extract captured hidden states → numpy arrays ─────────────────
        if (capture_hidden) {
            std::lock_guard<std::mutex> lk(capture_state_.mtx);
            for (auto& [layer_idx, cap] : capture_state_.captures) {
                if (cap.data.empty()) continue;

                // Create numpy array view over the captured data (copy to be safe)
                auto arr = py::array_t<float>({static_cast<ssize_t>(cap.n_embd)});
                auto buf = arr.request();
                std::memcpy(buf.ptr, cap.data.data(),
                            static_cast<size_t>(cap.n_embd) * sizeof(float));
                result.hidden_states[layer_idx] = std::move(arr);
            }
        }

        // ── Clear KV cache for next request (stateless mode) ─────────────
        llama_memory_clear(llama_get_memory(ctx_), true);

        result.text = std::move(full_text);
        result.completion_tokens = n_generated;
        return result;
    }
};

// ── pybind11 module ───────────────────────────────────────────────────────────

PYBIND11_MODULE(gaia_cpp, m) {
    m.doc() = "gaia_cpp — in-process llama.cpp backend for GAIA Engine";

    py::class_<GenerateResult>(m, "GenerateResult")
        .def_readonly("text",              &GenerateResult::text)
        .def_readonly("prompt_tokens",     &GenerateResult::prompt_tokens)
        .def_readonly("completion_tokens", &GenerateResult::completion_tokens)
        .def_readonly("hidden_states",     &GenerateResult::hidden_states);

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
                // Release GIL for C++ inference (may take seconds)
                py::gil_scoped_release release;
                return self.generate(prompt, max_tokens, temperature, top_p, top_k, capture_hidden);
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
                // Wrap the Python callable so it re-acquires the GIL
                // before calling back into Python, while the main inference
                // loop runs with the GIL released.
                auto cpp_callback = [&token_callback](const std::string& delta) {
                    py::gil_scoped_acquire acquire;
                    token_callback(delta);
                };
                py::gil_scoped_release release;
                return self.generate_stream(
                    prompt, cpp_callback, max_tokens, temperature, top_p, top_k, capture_hidden);
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

        .def("n_vocab",         &LlamaCppBackend::n_vocab)
        .def("n_embd",          &LlamaCppBackend::n_embd)
        .def("n_layer",         &LlamaCppBackend::n_layer)
        .def("has_gpu",         &LlamaCppBackend::has_gpu)
        .def("model_path",      &LlamaCppBackend::model_path)
        .def("capture_layers",  &LlamaCppBackend::capture_layers)
        .def("set_capture_layers", &LlamaCppBackend::set_capture_layers);
}
