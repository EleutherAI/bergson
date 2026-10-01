"""Does a gradient w.r.t. a non-parameter input survive torch.compile?

bergson reads its attribution scores from the gradient with respect to the
per-document weights (``cli.py``: ``scores = bwd_state.weight_grads``). Under
compile that gradient came back ``None``. This isolates whether that is a
general property of compile plus ``create_graph=True`` or something specific
to how the weights enter bergson's graph.
"""

import torch
import torch._functorch.config as functorch_config


def weighted_loss(x, w):
    return (x.pow(2) * w).sum()


def first_order(fn, tag, device):
    x = torch.randn(4, device=device, requires_grad=True)
    w = torch.ones(4, device=device, requires_grad=True)
    out = fn(x, w)
    gx, gw = torch.autograd.grad(out, [x, w], create_graph=True, allow_unused=True)
    print(
        f"  {tag:26s} grad_x={'None' if gx is None else tuple(gx.shape)}  "
        f"grad_w={'None' if gw is None else tuple(gw.shape)}",
        flush=True,
    )
    return gx, gw


def second_order(fn, tag, device):
    x = torch.randn(4, device=device, requires_grad=True)
    w = torch.ones(4, device=device, requires_grad=True)
    out = fn(x, w)
    gx, _ = torch.autograd.grad(out, [x, w], create_graph=True, allow_unused=True)
    (d2,) = torch.autograd.grad(gx.sum(), [w], allow_unused=True)
    print(
        f"  {tag:26s} d(grad_x)/dw={'None' if d2 is None else tuple(d2.shape)}",
        flush=True,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} on {device}", flush=True)

    print("first order (grad wrt both inputs):", flush=True)
    first_order(weighted_loss, "eager", device)
    first_order(torch.compile(weighted_loss), "compiled", device)

    functorch_config.donated_buffer = False
    first_order(torch.compile(weighted_loss), "compiled, no donated_buf", device)

    print("second order (double backward):", flush=True)
    second_order(weighted_loss, "eager", device)
    second_order(torch.compile(weighted_loss), "compiled", device)


if __name__ == "__main__":
    main()
