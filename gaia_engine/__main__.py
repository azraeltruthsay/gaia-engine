"""
GAIA Engine CLI entrypoint.

Usage:
    # Direct mode — loads model in-process (existing behavior)
    python -m gaia_engine --model /path/to/model --port 8092

    # Managed mode — zero-GPU standby with subprocess isolation
    python -m gaia_engine --managed --port 8092
    # Then load model via: POST /model/load {"model": "/path/to/model"}
"""

import argparse
import sys


def main():
    p = argparse.ArgumentParser(description="GAIA Inference Engine")
    p.add_argument("--model", default="", help="Model path (required for direct mode)")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", default="reduce-overhead",
                   choices=["reduce-overhead", "max-autotune", "none"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--managed", action="store_true",
                   help="Start in managed mode — zero GPU, subprocess isolation")
    args = p.parse_args()

    if args.managed:
        # Managed mode: stdlib-only, no torch/transformers import
        from gaia_engine.manager import serve_managed
        serve_managed(port=args.port, host=args.host)
    else:
        # Direct mode: loads model in-process (original behavior)
        if not args.model:
            p.error("--model is required in direct mode (or use --managed)")
        from gaia_engine.core import serve
        serve(args.model, args.port, args.device, getattr(args, "compile"), args.host)


if __name__ == "__main__":
    main()
