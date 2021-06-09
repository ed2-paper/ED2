base_config = {
    'agent': '@spinup_bis.algos.tf2.SUNRISE',
    'total_steps': 3000000,
    'num_test_episodes': 30,
    'ac_kwargs': {
        'hidden_sizes': [256, 256],
        'activation': 'relu',
    },
    'ac_number': 5,
    'autotune_alpha': True,
    'beta_bernoulli': 1.,
}

params_grid = {
    'task': [
        'Hopper-v2',
        'Walker2d-v2',
        'Ant-v2',
        'Humanoid-v2',
    ],
    'seed': [42, 7, 224444444, 11, 14,
             13, 5, 509758253, 777, 6051995,
             817604, 759621, 469592, 681422, 662896,
             680578, 50728, 680595, 650678, 984230,
             420115, 487860, 234662, 753671, 709357,
             755288, 109482, 626151, 459560, 629937],
}
