"""`python -m sglang.launch_server`, with the glmup_gguf method registered first.

The entry point covers sglang's worker processes; this covers the parent,
which parses the checkpoint's quantization_config before plugins load.
"""

import runpy

from . import register

if __name__ == "__main__":
    register()
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)
