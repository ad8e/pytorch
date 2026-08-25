"""Repro: AOTAutogradCache serves a stale artifact for a checkpoint body.

Run on BASE (b7930e431f5)  -> run 2 is a cache HIT and returns the OLD gradient.
Run on fc28dc88cac or later -> run 1 BYPASSES, no poisoned entry, run 2 is correct.

    python repro_nested_cache_collision.py
"""
import torch
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config
from torch._inductor.utils import fresh_cache
from torch.utils.checkpoint import checkpoint

# State the FX graph cannot see: allow_in_graph means dynamo does NOT trace into
# _opaque, so the dynamo graph records only its qualified name. AOTAutograd later
# traces it for real and bakes in whatever SCALE is at that moment.
SCALE = 2.0


@torch._dynamo.allow_in_graph
def _opaque(x):
    return x * SCALE


def body(x):
    return _opaque(x)


def fn(x):
    # The checkpoint HOP puts _opaque inside a nested `wrap_body_0` GraphModule.
    # Before this fix, check_cacheable never looked inside it.
    return checkpoint(body, x, use_reentrant=False)


def run():
    torch._dynamo.reset()
    counters.clear()
    x = torch.ones(4, requires_grad=True)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    (grad,) = torch.autograd.grad(compiled(x).sum(), x)
    c = counters["aot_autograd"]
    return grad, dict(
        miss=c["autograd_cache_miss"],
        hit=c["autograd_cache_hit"],
        bypass=c["autograd_cache_bypass"],
    )


def main():
    global SCALE
    with fresh_cache(), inductor_config.patch(
        {"fx_graph_cache": True, "fx_graph_remote_cache": False}
    ), functorch_config.patch({"enable_autograd_cache": True}):
        SCALE = 2.0
        grad1, c1 = run()
        print(f"run 1  SCALE=2.0  grad={grad1.tolist()}  {c1}")

        SCALE = 3.0
        grad2, c2 = run()
        print(f"run 2  SCALE=3.0  grad={grad2.tolist()}  {c2}")

    expected = torch.full((4,), 3.0)
    if torch.equal(grad2, expected):
        print("\nPASS: run 2 gradient is 3.0 (correct).")
        print(f"      run 1 bypassed={bool(c1['bypass'])}, run 2 hit={c2['hit']}")
        return 0
    print(f"\nFAIL: run 2 gradient is {grad2.tolist()}, expected {expected.tolist()}.")
    print(f"      Cache hit={c2['hit']} served run 1's artifact; SCALE=3.0 never took effect.")
    print("      This is the unsound cache hit the nested-validation commit prevents.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
