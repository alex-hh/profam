In AlphaFold the diffusion loss is not computed directly on the predicted noise,
but on predicted coordinates. Here we try to derive the relationships from the paper.

N.B. in the paper they have a dimensionless variable $r$ which here I'm replacing with x.


$$
x_l^{out} = \sigma^2_{data} / (\sigma^2_{data} + \hat{t}^2) x^{noisy}_l + \frac{\sigma_{data} \hat{t} }{x^{update}_l}
$$

We can probably infer the predicted noise...

DDPM:

$$
x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

so

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha_t}}} = \frac{x_t}{\sqrt{\bar{\alpha}_t}} - (\sqrt{1/\bar{\alpha}_t - 1}) \epsilon
$$
This is equation 15 in Ho.
where $\bar{\alpha}_t=\prod_{s=1}^t \alpha_s \:$ $\alpha_t = 1 - \beta_t$
And in DDPM, $\beta_t$ = kt.

In guided diffusion it is the value calculated by predict_xstart_from_eps:

```
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
```



They say that the diffusion implementation is based on Karras, rather than Ho.

Karras writes all diffusion models in terms of a denoiser $D_\theta(x, \sigma)$:

$$D_\theta (x, \sigma) = c_{skip} (\sigma) x + c_{out} (\sigma) F_\theta (c_{in} (\sigma) x; c_{noise}(\sigma))$$

For their proposed diffusion framework ('EDM'):

$$c_{skip}(\sigma) = \frac{\sigma^2_{data} }{(\sigma^2 + \sigma^2_{data})}$$
$$c_{out}(\sigma) = \sigma \sigma_{data} / \sqrt{\sigma^2_{data} + \sigma^2} $$
$$c_{in} = \frac{1}{\sqrt{\sigma^2 + \sigma^2_{data}}}$$

So this maps on to the AlphaFold notation with $\hat{t} = \sigma$

'Where $\hat{t}$ is the sampled noise level, $\sigma_{data}$ is a constant determined by the variance of the data (set to 16)'

We minimise the MSE to $x_0$:

$$\|D(x_0 + n; \sigma) - x_0\|^2$$

Resulting in the following relation (i.e. the denoising network learns the gradient of the noised distribution.)

$$\nabla_{x} \text{log}p(x;\sigma) = \frac{(D(x;\sigma)-x)}{\sigma^2}$$

In the Karras notation, for DDPM:

$$c_{skip} = 1$$
$$c_{out} = -\sigma $$

I think to get full equivalence, you have to incorporate the expressions for sigma and for loss scaling, and perhaps preprocessing


MSE on $x_0$ vs MSE on $\epsilon$:
Karras supp. has extensive derivations; but they require ode and sde knowledge.
