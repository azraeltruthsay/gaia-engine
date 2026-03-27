cmake_minimum_required(VERSION 3.18)
project(gaia_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ──────────────────────────────────────────────────────────────────
# Installed via "pip install pybind11[global]" in the builder stage.
find_package(pybind11 CONFIG REQUIRED)

# ── llama.cpp paths ───────────────────────────────────────────────────────────
# These paths match the builder stage in Dockerfile.gaia-engine:
#   git clone --depth 1 --branch ${LLAMA_CPP_TAG} ... /build/llama.cpp
# Build output is at /build/llama.cpp/build/
# In llama.cpp b8250, all .so files land in build/bin/ alongside the executables.
set(LLAMA_SRC_DIR   "/build/llama.cpp"            CACHE PATH "llama.cpp source root")
set(LLAMA_BUILD_DIR "${LLAMA_SRC_DIR}/build"      CACHE PATH "llama.cpp cmake build dir")
set(LLAMA_BIN_DIR   "${LLAMA_BUILD_DIR}/bin"      CACHE PATH "llama.cpp build output (exes + .so files)")

set(LLAMA_INCLUDE_DIR "${LLAMA_SRC_DIR}/include"        CACHE PATH "llama.h location")
set(GGML_INCLUDE_DIR  "${LLAMA_SRC_DIR}/ggml/include"   CACHE PATH "ggml.h location")

# ── gaia_cpp extension ────────────────────────────────────────────────────────
pybind11_add_module(gaia_cpp
    gaia_cpp.cpp
)

target_include_directories(gaia_cpp PRIVATE
    ${LLAMA_INCLUDE_DIR}
    ${GGML_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}   # for hidden_state_capture.hpp
)

target_link_libraries(gaia_cpp PRIVATE
    ${LLAMA_BIN_DIR}/libllama.so
    ${LLAMA_BIN_DIR}/libggml.so
)

# RPATH: find libllama.so / libggml.so at /usr/local/lib in the runtime container.
# The Dockerfile copies those .so files there from the builder stage.
set_target_properties(gaia_cpp PROPERTIES
    INSTALL_RPATH "/usr/local/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Optimization flags
target_compile_options(gaia_cpp PRIVATE -O3 -march=native -ffast-math)

# ── Install ───────────────────────────────────────────────────────────────────
# The Dockerfile copies the built .so to the Python package location.
# This install target is optional — used if building outside Docker.
install(TARGETS gaia_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
