import torch
import torch.nn as nn


class AdaptiveTanh(nn.Module):
    """tanh(n * a * x) with a trainable slope a (Jagtap et al. 2020). Sharper fronts train faster."""

    def __init__(self, n=10.0):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return torch.tanh(self.n * self.a * x)


class WedgePINN(nn.Module):
    """MLP (x, y) -> (u, v, rho, p) [+ n_extra raw outputs, e.g. a local viscosity].

    param = "freestream": outputs are a correction to the free stream (u = u_inf + o, ...),
            with the last layer zero-initialised the network starts at the free stream.
    param = "raw":        u = o0, v = o1 with default (random) initialisation, as in the old code.
    rho and p are always exp(.) of an output, so they stay positive.
    """

    def __init__(self, bounds, freestream, width=64, depth=6, adaptive=True, out_scale=(1.0, 1.0, 1.0, 1.0),
                 param="freestream", n_extra=0, normalize_inputs=True):
        super().__init__()
        (x0, x1), (y0, y1) = bounds
        if not normalize_inputs:          # identity map: z = X
            x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0
        self.register_buffer("lo", torch.tensor([x0, y0], dtype=torch.float32))
        self.register_buffer("hi", torch.tensor([x1, y1], dtype=torch.float32))
        self.register_buffer("q_inf", torch.tensor(freestream.state(), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(list(out_scale) + [1.0] * n_extra, dtype=torch.float32))
        self.param = param
        self.n_extra = n_extra

        layers = []
        d_in = 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), AdaptiveTanh() if adaptive else nn.Tanh()]
            d_in = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 4 + n_extra)
        if param == "freestream":
            for m in self.body:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.extra = None   # raw extra outputs of the last call

    def _states(self, o):
        if self.param == "freestream":
            u = self.q_inf[0] + o[..., 0]
            v = self.q_inf[1] + o[..., 1]
            r = self.q_inf[2] * torch.exp(o[..., 2])
            p = self.q_inf[3] * torch.exp(o[..., 3])
        else:
            u, v, r, p = o[..., 0], o[..., 1], torch.exp(o[..., 2]), torch.exp(o[..., 3])
        return u, v, r, p

    def forward(self, X):
        z = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        o = self.head(self.body(z)) * self.scale
        self.extra = o[:, 4:]
        return torch.stack(self._states(o), dim=1)

    def forward_derivs(self, X, second=True):
        """q and its spatial derivatives in one forward sweep (Taylor-mode, exact).

        Returns q (N,4), dq (2,N,4) = (q_x, q_y) and, if second, d2q (2,N,4) = (q_xx, q_yy).
        The raw extra outputs (values only) are left in self.extra.
        Much cheaper than repeated autograd.grad calls, and still differentiable w.r.t. the
        network parameters by ordinary backprop.
        """
        N = X.shape[0]
        sc = 2.0 / (self.hi - self.lo)                          # dz/dX per input dim
        z = sc * (X - self.lo) - 1.0
        # derivative of the input features: direction k only moves feature k
        dz = torch.zeros(2, N, 2, device=X.device, dtype=X.dtype)
        dz[0, :, 0] = sc[0]
        dz[1, :, 1] = sc[1]
        d2z = torch.zeros_like(dz) if second else None
        h, dh, d2h = z, dz, d2z
        mods = list(self.body)
        for i in range(0, len(mods), 2):
            lin, act = mods[i], mods[i + 1]
            a = lin(h)
            da = dh @ lin.weight.T
            d2a = d2h @ lin.weight.T if second else None
            s = act.n * act.a if isinstance(act, AdaptiveTanh) else 1.0
            h = torch.tanh(s * a)
            g = s * (1.0 - h * h)                                # d tanh(s a) / da
            dh = g * da
            if second:
                d2h = g * d2a - 2.0 * s * h * g * da * da
        o = self.head(h) * self.scale
        do = (dh @ self.head.weight.T) * self.scale
        d2o = (d2h @ self.head.weight.T) * self.scale if second else None
        self.extra = o[:, 4:]
        self.extra_d = do[..., 4:]                               # (2, N, n_extra): d(extra)/dx, d(extra)/dy

        u, v, r, p = self._states(o)
        q = torch.stack([u, v, r, p], dim=1)
        dq = torch.stack([do[..., 0], do[..., 1], r * do[..., 2], p * do[..., 3]], dim=-1)
        if not second:
            return q, dq, None
        d2q = torch.stack([d2o[..., 0], d2o[..., 1],
                           r * (d2o[..., 2] + do[..., 2] ** 2),
                           p * (d2o[..., 3] + do[..., 3] ** 2)], dim=-1)
        return q, dq, d2q
