import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from jax import vmap
from bsm.bayesian_regression.gaussian_processes.kernels import Kernel
from jaxtyping import PyTree, Float, Array, Scalar, Key
from jax.nn import softplus
from bsm.utils.normalization import Data


class ARD(Kernel):
    def __init__(self, input_dim: int, length_scale: float | None = None):
        super().__init__(input_dim)
        self.length_scale = length_scale

    def _apply(self,
               x1: Float[Array, 'input_dim'],
               x2: Float[Array, 'input_dim'],
               kernel_params: PyTree) -> Scalar:
        pseudo_length_scale = kernel_params['pseudo_length_scale']
        length_scale = softplus(pseudo_length_scale)
        assert pseudo_length_scale.shape == (self.input_dim,)
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2 / length_scale ** 2))

    def init(self, key: Key[Array, '2']) -> PyTree:
        if self.length_scale is None:
            length_scale = jr.uniform(key=key, minval=0.1, maxval=1.0, shape=(self.input_dim,))
            pseudo_length_scale = jnp.log(jnp.exp(length_scale) - 1)
            return {'pseudo_length_scale': pseudo_length_scale}
        else:
            assert self.length_scale > 0
            length_scale = jnp.ones(self.input_dim) * self.length_scale
            pseudo_length_scale = jnp.log(jnp.exp(length_scale) - 1)
            return {'pseudo_length_scale': pseudo_length_scale}


if __name__ == '__main__':
    from bsm.bayesian_regression.gaussian_processes import GaussianProcess
    from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
    import time
    import matplotlib.pyplot as plt

    # import jax
    # jax.config.update('jax_log_compiles', True)

    key = jr.PRNGKey(0)
    example = '1d_to_2d'  # or '2d_to_3d'
    if example == '1d_to_2d':
        input_dim = 1
        output_dim = 2

        noise_level = 0.1
        d_l, d_u = 0, 10
        xs = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
        ys = jnp.concatenate([jnp.sin(2 * xs) + jnp.sin(3 * xs), jnp.cos(3 * xs)], axis=1)
        ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
        data_std = noise_level * jnp.ones(shape=(output_dim,))
        data = Data(inputs=xs, outputs=ys)

    elif example == '2d_to_3d':
        input_dim = 2
        output_dim = 3

        noise_level = 0.1
        key, subkey = jr.split(key)
        xs = jr.uniform(key=subkey, shape=(64, input_dim))
        ys = jnp.stack([jnp.sin(xs[:, 0]),
                        jnp.cos(3 * xs[:, 1]),
                        jnp.cos(xs[:, 1]) * jnp.sin(xs[:, 1])], axis=1)
        ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
        data_std = noise_level * jnp.ones(shape=(output_dim,))
        data = Data(inputs=xs, outputs=ys)

    logging = False
    num_particles = 10
    model = GPStatisticalModel(
        kernel=ARD(input_dim=input_dim),
        input_dim=input_dim,
        output_dim=output_dim,
        output_stds=data_std,
        logging_wandb=False,
        f_norm_bound=3 * jnp.array([2.0, 3.0, ]),
        beta=None,
        num_training_steps=optax.constant_schedule(1000)
    )
    model_state = model.init(jr.PRNGKey(0))
    start_time = time.time()
    print('Starting with training')
    if logging:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    # model_state = model.update(data=data, model_state=model_state)
    print(f'Training time: {time.time() - start_time:.2f} seconds')
    model_state = model.update(data=data, stats_model_state=model_state)

    if example == '1d_to_2d':
        test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
        preds = model.predict_batch(test_xs, model_state)
        test_ys = jnp.concatenate([jnp.sin(2 * test_xs) + jnp.sin(3 * test_xs), jnp.cos(3 * test_xs)], axis=1)

        for j in range(output_dim):
            plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
            plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
            plt.fill_between(test_xs.reshape(-1),
                             (preds.mean[:, j] - preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
                                                                                         j]).reshape(-1),
                             (preds.mean[:, j] + preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
                                                                                         j]).reshape(
                                 -1),
                             label=r'$2\sigma$', alpha=0.3, color='blue')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.savefig(f'gp_{j}.pdf')
            plt.show()
