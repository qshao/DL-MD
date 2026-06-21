import torch
from lsmd import batching
from lsmd import transfer_model as tm
from lsmd.model import NoiseSchedule


def _toy_graph(n, k=4, tau=100.0):
    e = n * k
    return {
        "node_feats": torch.randn(n, 24),
        "edge_index": torch.randint(0, n, (2, e)),
        "edge_feats": torch.randn(e, 13),
        "u_target": torch.randn(n, 6),
        "tau": tau,
    }


def test_encoder_context_shape():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    assert ctx.shape == (13, 32)


def test_encoder_graphs_do_not_interact():
    torch.manual_seed(0)
    g0, g1 = _toy_graph(5, tau=100.0), _toy_graph(8, tau=100.0)
    enc = tm.StructuralEncoder(hidden=32, layers=2).eval()
    solo = batching.union_collate([g0])
    pair = batching.union_collate([g0, g1])
    with torch.no_grad():
        c_solo = enc(solo["node_feats"], solo["edge_index"], solo["edge_feats"],
                     solo["tau"], solo["batch"])
        c_pair = enc(pair["node_feats"], pair["edge_index"], pair["edge_feats"],
                     pair["tau"], pair["batch"])
    assert torch.allclose(c_solo, c_pair[:5], atol=1e-5)


def test_denoiser_output_shape_default_one_layer():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    den = tm.Denoiser(hidden=32, n_denoise_layers=1)
    s = torch.rand(2)
    out = den(u["u_target"], s, ctx, u["edge_index"], u["edge_feats"], u["batch"])
    assert out.shape == (13, 6)


def test_denoiser_zero_layers_is_pure_mlp():
    u = batching.union_collate([_toy_graph(6)])
    enc = tm.StructuralEncoder(hidden=32, layers=2)
    ctx = enc(u["node_feats"], u["edge_index"], u["edge_feats"], u["tau"], u["batch"])
    den = tm.Denoiser(hidden=32, n_denoise_layers=0)
    out = den(u["u_target"], torch.rand(1), ctx, u["edge_index"],
              u["edge_feats"], u["batch"])
    assert out.shape == (6, 6)


def test_forward_equals_encode_then_denoise():
    torch.manual_seed(0)
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.CachedPropagator(hidden=32, layers=2, n_denoise_layers=1).eval()
    s = torch.rand(2)
    with torch.no_grad():
        direct = net(u["u_target"], s, u["node_feats"], u["edge_index"],
                     u["edge_feats"], u["tau"], u["batch"])
        ctx = net.encode(u["node_feats"], u["edge_index"], u["edge_feats"],
                         u["tau"], u["batch"])
        split = net.denoise(u["u_target"], s, ctx, u["edge_index"],
                            u["edge_feats"], u["batch"])
    assert torch.allclose(direct, split, atol=1e-6)


def test_cached_propagator_is_drop_in_for_union_loss():
    u = batching.union_collate([_toy_graph(5), _toy_graph(8)])
    net = tm.CachedPropagator(hidden=32, layers=2)
    sched = NoiseSchedule(T=50)
    loss = tm.ddpm_loss_union(net, u["u_target"], u["node_feats"],
                              u["edge_index"], u["edge_feats"], u["tau"],
                              u["batch"], sched)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_cached_sampler_shape_and_finite():
    u = batching.union_collate([_toy_graph(7)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)
    out = tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                      u["edge_feats"], u["tau"], u["batch"],
                                      sched, steps=5)
    assert out.shape == (7, 6)
    assert torch.isfinite(out).all()


def test_cached_sampler_matches_uncached_reference():
    u = batching.union_collate([_toy_graph(6)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    torch.manual_seed(123)
    cached = tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                         u["edge_feats"], u["tau"], u["batch"],
                                         sched, steps=5, eta=0.0)

    torch.manual_seed(123)
    T = sched.T
    N = u["node_feats"].shape[0]
    uu = torch.randn(N, net.point_dim)
    t_full = torch.round(torch.linspace(T - 1, 0, 6)).long().clamp(0, T - 1)
    with torch.no_grad():
        for i in range(5):
            t = t_full[i].item(); t_prev = t_full[i + 1].item()
            s = torch.full((u["tau"].shape[0],), t / T)
            eps = net(uu, s, u["node_feats"], u["edge_index"], u["edge_feats"],
                      u["tau"], u["batch"])
            sqrt_ab_t = sched.sqrt_alphas_bar[t]
            sqrt_1mab_t = sched.sqrt_one_minus_alphas_bar[t]
            ab_prev = sched.alphas_bar[t_prev]; ab_t = sched.alphas_bar[t]
            u0 = (uu - sqrt_1mab_t * eps) / sqrt_ab_t.clamp_min(1e-8)
            dir_coeff = (1 - ab_prev).clamp_min(0.0).sqrt()
            uu = ab_prev.sqrt() * u0 + dir_coeff * eps
    assert torch.allclose(cached, uu, atol=1e-5)


def test_cached_sampler_encodes_once_per_propagation_step():
    u = batching.union_collate([_toy_graph(6)])
    net = tm.CachedPropagator(hidden=32, layers=2).eval()
    sched = NoiseSchedule(T=50)

    calls = {"encode": 0, "denoise": 0}
    real_encode, real_denoise = net.encode, net.denoise
    net.encode = lambda *a, **k: (calls.__setitem__("encode", calls["encode"] + 1) or real_encode(*a, **k))
    net.denoise = lambda *a, **k: (calls.__setitem__("denoise", calls["denoise"] + 1) or real_denoise(*a, **k))

    tm.sample_ddpm_union_cached(net, u["node_feats"], u["edge_index"],
                                u["edge_feats"], u["tau"], u["batch"],
                                sched, steps=5)
    assert calls["encode"] == 1
    assert calls["denoise"] == 5
