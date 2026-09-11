import torch

from bergson.magic.data_stream import DataStream, pad_dataset_to_batch_size
from bergson.validate import mean_query_loss

from .test_ddp import _make_dataset, _make_model


def test_mean_query_loss_skips_all_padding_micro_batches():
    """A plain model returns NaN on a micro-batch with no supervised tokens,
    so padding rows split into their own micro-batches must not poison the
    mean: the padded stream gives the single document's loss."""
    model = _make_model().eval()
    docs = _make_dataset().select(range(1))
    padded, n, _, weight_pad = pad_dataset_to_batch_size(docs, 4, 1, "Q", 0)
    stream = DataStream(padded, 4, device="cpu", weight_shape=(n,))
    stream.weights.data[-weight_pad:] = 0.0
    with torch.no_grad():
        padded_loss = mean_query_loss(model, stream, grad_accum_steps=4)
        ref = mean_query_loss(
            model, DataStream(docs, 1, device="cpu", weight_shape=(1,))
        )
    assert torch.isfinite(padded_loss)
    torch.testing.assert_close(padded_loss, ref, rtol=1e-4, atol=1e-5)
