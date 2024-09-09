import argparse
ENTITY = 'cbiel01'

def experiment(project_name: str = 'Offline_Smoother_Comparison',
               environment: str = 'pendulum',
               differentiator: str = 'BNNSmoother',
               num_offline_samples: int = 2000,
               num_online_samples: int = 200,
               noise_level: float | list | bool = False,
               seed : int = 42,
               smoother_steps: int = 16_000,
               smoother_features: tuple = (64, 64),
               smoother_reg_type: str = 'first',
               smoother_lambda: float = 1e-4,
               smoother_degree: int = 15,
               measurement_dt_ratio: int = 1,
               save_plots: bool = True,
               save_data: bool = False,
            ):
    
    import jax
    import jax.numpy as jnp
    import chex
    import jax.random as jr

    from mbrl.envs.pendulum_ct import ContinuousPendulumEnv
    from mbrl.envs.bicyclecar import BicycleEnv
    from mbrl.envs.cartpole import ContinuousCartpoleEnv

    from mbrl.utils.offline_data import DifferentiatorOfflineData, save_transitions

    assert environment in ['pendulum', 'bicycle', 'cartpole']
    assert differentiator in ['BNNSmoother', 'NumSmoother', 'PolSmoother']
    assert smoother_reg_type in ['zero', 'first', 'second']

    noise_key, key = jr.split(jr.PRNGKey(seed), 2)
    # Create the environments with the correct arguments
    if environment == 'pendulum':
        if isinstance(noise_level, float):
            noise_level = jnp.array([0.05, 0.1]) * noise_level
        elif isinstance(noise_level, bool):
            if noise_level:
                noise_level = jnp.array([0.05, 0.1])
            else:
                noise_level = jnp.array([0.0, 0.0])
        elif isinstance(noise_level, list):
            noise_level = jnp.array(noise_level)
            chex.assert_shape(noise_level, (2,))
        env = ContinuousPendulumEnv(noise_level=noise_level,
                                    init_noise_key=noise_key)
        init_state_range = jnp.array([[-1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
        
    elif environment == 'cartpole':
        if isinstance(noise_level, float):
            noise_level = jnp.array([0.2, 0.05, 0.1, 0.1]) * noise_level
        elif isinstance(noise_level, bool):
            if noise_level:
                noise_level = jnp.array([0.2, 0.05, 0.1, 0.1])
            else:
                noise_level = jnp.array([0.0, 0.0, 0.0, 0.0])
        elif isinstance(noise_level, list):
            noise_level = jnp.array(noise_level)
            chex.assert_shape(noise_level, (4,))
        env = ContinuousCartpoleEnv(noise_level=noise_level,
                                    init_noise_key=noise_key)
        init_state_range = jnp.array([[-1.0, -1.0, 0.0, -1.0, -1.0], [1.0, -1.0, 0.0, 1.0, 1.0]])
        
    elif environment == 'bicycle':
        if isinstance(noise_level, float):
            use_obs_noise = True if noise_level > 0 else False
        elif isinstance(noise_level, bool):
            use_obs_noise = noise_level
        elif isinstance(noise_level, list):
            noise_level = jnp.array(noise_level)
            use_obs_noise = jnp.any(noise_level > 0)
        env = BicycleEnv(init_noise_key=noise_key,
                         use_obs_noise=use_obs_noise)
        init_state_range = jnp.concatenate([env.reset().pipeline_state.reshape(1, -1),
                                            env.reset().pipeline_state.reshape(1, -1)], axis=0)
    else:
        raise NotImplementedError
    
    # Generate the differentiator
    if differentiator == 'BNNSmoother':
        from diff_smoothers.BNN_Differentiator import BNNSmootherDifferentiator as BNN_Smoother
        from bsm.bayesian_regression import DeterministicEnsemble

        smoother = BNN_Smoother(state_dim=env.observation_size,
                                output_stds=jnp.ones(shape=(env.observation_size,)) * 0.1,
                                logging_wandb=False,
                                beta=jnp.ones(shape=(env.observation_size,))*2.0,
                                num_particles=5,
                                features=smoother_features,
                                bnn_type=DeterministicEnsemble,
                                train_share=1.0,
                                num_training_steps=smoother_steps,
                                weight_decay=smoother_lambda,
                                return_best_model=True,
                                eval_frequency=1000,)
    elif differentiator == 'NumSmoother':
        from diff_smoothers.Numerical_Differentiator import TikhonovDifferentiator as Num_Smoother

        smoother = Num_Smoother(state_dim=env.observation_size,
                                regtype=smoother_reg_type,
                                lambda_=smoother_lambda)
        
    elif differentiator == 'PolSmoother':
        from diff_smoothers.PolFit_Differentiator import PolFit_Differentiator as Pol_Smoother
        
        smoother = Pol_Smoother(state_dim=env.observation_size,
                                degree=smoother_degree,
                                lambda_=smoother_lambda)
        
    else:
        raise NotImplementedError
    
    # Generate the offline data with the smoothers
    offline_data_gen = DifferentiatorOfflineData(differentiator=smoother,
                                                 env=env,
                                                 init_state_range=init_state_range)
    if save_data or save_plots:
        import os
        filename = f'./results/smoother_eval/{project_name}/{differentiator}_{environment}'
        if not os.path.exists(f'./results/smoother_eval/{project_name}/'):
            if not os.path.exists('./results/smoother_eval/'):
                if not os.path.exists('./results/'):
                    os.mkdir('./results/')
                os.mkdir('./results/smoother_eval/')
            os.mkdir(f'./results/smoother_eval/{project_name}/')
        

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=num_offline_samples,
                                                       trajectory_length=num_online_samples,
                                                       plot_results=save_plots,
                                                       measurement_dt_ratio=measurement_dt_ratio,
                                                       state_data_source='smoother',
                                                       pathname=filename)
    if save_data:
        
        # Check if the file already exists
        if os.path.exists(filename):
            i = 1
            while os.path.exists(f'{filename}_{i}'):
                i += 1
            filename = f'{filename}_{i}'
        save_transitions(offline_data, f'{filename}.npy')
    



def main(args):
    experiment(project_name=args.project_name,
               environment=args.environment,
               differentiator=args.differentiator,
               num_offline_samples=args.num_offline_samples,
               num_online_samples=args.num_online_samples,
               noise_level=args.noise_level,
               seed=args.seed,
               smoother_steps=args.smoother_steps,
               smoother_features=args.smoother_features,
               smoother_reg_type=args.smoother_reg_type,
               smoother_lambda=args.smoother_lambda,
               smoother_degree=args.smoother_degree,
               measurement_dt_ratio=args.measurement_dt_ratio,
               save_plots=args.save_plots,
               save_data=args.save_data,
            )
    
if __name__ == '__main__':

    def underscore_to_tuple(value: str):
        return tuple(map(int, value.split('_')))
    
    def underscore_to_list(value: str):
        return list(map(float, value.split('_')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='basic_comp')
    parser.add_argument('--environment', type=str, default='pendulum')
    parser.add_argument('--differentiator', type=str, default='NumSmoother')
    parser.add_argument('--num_offline_samples', type=int, default=600)
    parser.add_argument('--num_online_samples', type=int, default=200)
    parser.add_argument('--noise_level', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoother_steps', type=int, default=64_000)
    parser.add_argument('--smoother_features', type=underscore_to_tuple, default='64_64_64')
    parser.add_argument('--smoother_reg_type', type=str, default='first')
    parser.add_argument('--smoother_lambda', type=float, default=1e-4)
    parser.add_argument('--smoother_degree', type=int, default=15)
    parser.add_argument('--measurement_dt_ratio', type=int, default=1)
    parser.add_argument('--save_plots', type=bool, default=True)
    parser.add_argument('--save_data', type=bool, default=False)
    
    args = parser.parse_args()
    main(args)