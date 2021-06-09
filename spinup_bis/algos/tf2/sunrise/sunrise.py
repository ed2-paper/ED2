"""SUNRISE algorithm implementation."""

import os
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from spinup_bis.algos.tf2.sunrise import core
from spinup_bis.utils import logx


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SUNRISE agents."""

    def __init__(self, obs_dim, act_dim, size, ac_number):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.mask_buf = np.zeros([size, ac_number], dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = size
        self.ac_number = ac_number


    def store(self, obs, act, rew, next_obs, done, mask):
        """Store the transitions and masks in the replay buffer."""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.mask_buf[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        """Sample batch of buffered experience."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs_shape = [self.ac_number, batch_size, self.obs_dim]
        act_shape = [self.ac_number, batch_size, self.act_dim]
        rew_shape = [self.ac_number, batch_size]

        obs1 = np.broadcast_to(self.obs1_buf[idxs], obs_shape)
        obs2 = np.broadcast_to(self.obs2_buf[idxs], obs_shape)
        acts = np.broadcast_to(self.acts_buf[idxs], act_shape)
        rews = np.broadcast_to(self.rews_buf[idxs], rew_shape)
        done = np.broadcast_to(self.done_buf[idxs], rew_shape)
        masks = tf.transpose(self.mask_buf[idxs])

        return dict(
            obs1=tf.convert_to_tensor(obs1),
            obs2=tf.convert_to_tensor(obs2),
            acts=tf.convert_to_tensor(acts),
            rews=tf.convert_to_tensor(rews),
            done=tf.convert_to_tensor(done),
            masks=tf.convert_to_tensor(masks),
        )


def sunrise(
    env_fn,
    actor_critic=core.MLPActorCriticFactory,
    ac_kwargs=None,
    ac_number=1,
    seed=0,
    total_steps=1_000_000,
    log_every=10_000,
    replay_size=1_000_000,
    gamma=0.99,
    polyak=0.995,
    lr=0.001,
    alpha=0.2,
    batch_size=256,
    start_steps=10_000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kwargs=None,
    save_freq=10_000,
    save_path=None,
    autotune_alpha=False,
    target_entropy=None,
    alpha_lr=3e-4,
    use_weighted_bellman_backup=True,
    bellman_temp=10,
    ucb_lambda=1,
    beta_bernoulli=0.5,
):
    """SUNRISE

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` kwargs
            and returns actor and critic tf.keras.Model-s.

            Actor should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ===========  ================  =====================================

            Critic should take an observation and action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``q``        (batch,)          | Gives the current estimate of Q*
                                           | state and action in the input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SUNRISE.

        ac_number (int): Number of the actor-critic models in the ensemble.

        seed (int): Seed for random number generators.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of environment iterations) to save
            the current policy and value function.

        save_path (str): The path specifying where to save the trained actor
            model. Setting the value to None turns off the saving.

        autotune_alpha (bool): Tune alpha automatically to achieve certain
            entropy of policy.

        target_entropy (Optional[float]): If not none optimize alpha parameter
            to encourage actor to achieve selected entropy.

        alpha_lr (float): Learning rate of alpha optimizer. Has effect only when
            autotune_alpha is not True.

        use_weighted_bellman_backup (bool): Whether the Bellman backup should
            be reweighted based on the critic ensemble disagreement.

        bellman_temp (float): Temperature parameter used in calculating the
            weight for the weighted Bellman backup.

        ucb_lambda (float): Weight of the standard deviation part
            of the UCB exploration scoring formula.

        beta_bernoulli (float): Probability of drawing 1 in the Bernoulli
            distribution, used for generating masks for the bootstrap.
    """
    pwd = os.getcwd()  # pylint: disable=possibly-unused-variable
    logger = logx.EpochLogger(**(logger_kwargs or {}))
    logger.save_config(locals())

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # This implementation assumes all dimensions share the same bound!
    assert np.all(env.action_space.high == env.action_space.high[0])

    # Share information about observation and action spaces with policy.
    ac_kwargs = ac_kwargs or {}
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['ac_number'] = ac_number

    # Network
    ac_factory = actor_critic(**ac_kwargs)

    actor = ac_factory.make_actor()
    actor.build(input_shape=(None, obs_dim))

    critic1 = ac_factory.make_critic()
    critic1.build(input_shape=(None, obs_dim + act_dim))
    critic2 = ac_factory.make_critic()
    critic2.build(input_shape=(None, obs_dim + act_dim))

    critic_variables = critic1.trainable_variables + critic2.trainable_variables

    # Target networks
    target_critic1 = ac_factory.make_critic()
    target_critic1.build(input_shape=(None, obs_dim + act_dim))
    target_critic2 = ac_factory.make_critic()
    target_critic2.build(input_shape=(None, obs_dim + act_dim))

    # Copy weights
    target_critic1.set_weights(critic1.get_weights())
    target_critic2.set_weights(critic2.get_weights())

    # Experience buffer.
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        size=replay_size,
        ac_number=ac_number,
    )

    # Bernoulli distribution used for generating masks
    bernoulli_distr = tfp.distributions.Bernoulli(probs=beta_bernoulli)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if autotune_alpha:
        if target_entropy is None:
            target_entropy = core.heuristic_target_entropy(env.action_space)
        alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)
        log_alpha = tf.Variable(tf.math.log(alpha))
        alpha = tfp.util.DeferredTensor(log_alpha, tf.math.exp)

    @tf.function
    def evaluation_policy(obs):
        obs = tf.broadcast_to(obs, [ac_number, 1, *obs.shape])
        mu, _, _ = actor(obs)
        act = tf.reduce_mean(mu, axis=0)
        return act[0]

    @tf.function
    def behavioural_policy(obs):
        obs_actor = tf.broadcast_to(obs, [ac_number, 1, *obs.shape])
        obs_critic = tf.broadcast_to(obs, [ac_number, ac_number, *obs.shape])

        # Take the action
        _, pi, _ = actor(obs_actor)
        act = tf.reshape(pi, [1, ac_number, *pi.shape[2:]])
        act = tf.broadcast_to(act, [ac_number, ac_number, *pi.shape[2:]])

        # Evaluate the actions through the critic ensemble
        qs = 0.5 * critic1([obs_critic, act]) + 0.5 * critic2([obs_critic, act])
        q_mean = tf.reduce_mean(qs, axis=0)
        q_std = tf.math.reduce_std(qs, axis=0)
        scores = q_mean + ucb_lambda * q_std

        # Return the action with the highest score
        return pi[tf.math.argmax(scores)][0]

    @tf.function
    def sample_mask():
        return bernoulli_distr.sample(sample_shape=(ac_number,))

    @tf.function
    def update_alpha(obs):
        _, _, logp_pi = actor(obs)  # batch['obs1']

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * alpha * \
                           tf.stop_gradient(logp_pi + target_entropy)
            alpha_loss = tf.reduce_mean(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [log_alpha])
        alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [log_alpha]))

        return dict(
            AlphaLoss=alpha_loss,
            ActorEntropy=tf.reduce_mean(-logp_pi),
        )

    @tf.function
    def learn_on_batch(obs1, obs2, acts, rews, done, masks):
        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph.
            _, pi, logp_pi = actor(obs1)
            q1 = critic1([obs1, acts])
            q2 = critic2([obs1, acts])

            # Compose q with pi, for pi-learning.
            q1_pi = critic1([obs1, pi])
            q2_pi = critic2([obs1, pi])

            # Get actions and log probs of actions for next states.
            _, pi_next, logp_pi_next = actor(obs2)

            # Target Q-values, using actions from *current* policy.
            target_q1 = target_critic1([obs2, pi_next])
            target_q2 = target_critic2([obs2, pi_next])

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions.
            # Using Clipped Double-Q targets.
            q_backup = tf.stop_gradient(rews + gamma * (1 - done) * (
                min_target_q - alpha * logp_pi_next))

            # Actor loss
            pi_loss = tf.reduce_mean(
                masks * (alpha * logp_pi - min_q_pi)
            )

            # Critics loss
            q1_loss = 0.5 * ((q_backup - q1) ** 2)
            q2_loss = 0.5 * ((q_backup - q2) ** 2)

            if use_weighted_bellman_backup:
                q_std = tf.math.reduce_std(min_target_q, axis=0)
                q_weight = tf.math.sigmoid(-q_std * bellman_temp) + 0.5

                value_loss = tf.reduce_mean(
                    q_weight * tf.reduce_mean(
                        masks * (q1_loss + q2_loss),
                        axis=0,
                    )
                )
            else:
                value_loss = tf.reduce_mean(masks * (q1_loss + q2_loss))

        # Compute gradients and do updates.
        actor_gradients = g.gradient(pi_loss, actor.trainable_variables)
        optimizer.apply_gradients(
            zip(actor_gradients, actor.trainable_variables))
        critic_gradients = g.gradient(value_loss, critic_variables)
        optimizer.apply_gradients(
            zip(critic_gradients, critic_variables))

        # Polyak averaging for target variables.
        for v, target_v in zip(critic1.trainable_variables,
                               target_critic1.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic2.trainable_variables,
                               target_critic2.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

        del g
        return dict(
            pi_loss=pi_loss,
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q1=q1,
            q2=q2,
            logp_pi=logp_pi,
        )

    def test_agent():
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len, task_ret = test_env.reset(), False, 0, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time.
                o, r, d, info = test_env.step(
                    evaluation_policy(tf.convert_to_tensor(o)))
                ep_ret += r
                ep_len += 1
                task_ret += info.get('reward_task', 0)
            logger.store(TestEpRet=ep_ret,
                         TestEpLen=ep_len,
                         TestTaskRet=task_ret,
                         TestTaskSolved=info.get('is_solved', False))

    start_time = time.time()
    iter_begin_time = start_time
    o, ep_ret, ep_len, task_ret = env.reset(), 0, 0, 0
    # Main loop: collect experience in env and update/log each epoch.
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = behavioural_policy(tf.convert_to_tensor(o))
        else:
            a = env.action_space.sample()

        # Step the environment.
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1
        task_ret += info.get('reward_task', 0)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state).
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer.
        mask = sample_mask()
        replay_buffer.store(o, a, r, o2, d, mask)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling.
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret,
                         EpLen=ep_len,
                         TaskRet=task_ret,
                         TaskSolved=info.get('is_solved', False))
            o, ep_ret, ep_len, task_ret = env.reset(), 0, 0, 0

        # Update handling.
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                results = learn_on_batch(**batch)
                metrics = dict(
                    LossPi=results['pi_loss'],
                    LossQ1=results['q1_loss'],
                    LossQ2=results['q2_loss'],
                    LogPi=results['logp_pi'],
                )

                for idx, (q1, q2) in enumerate(
                        zip(results['q1'], results['q2'])):
                    metrics.update({
                        f'Q1Vals_{idx + 1}': q1,
                        f'Q2Vals_{idx + 1}': q2,
                        f'QDiff_{idx + 1}': np.abs(q1 - q2),
                    })
                logger.store(**metrics)

                if autotune_alpha:
                    results = update_alpha(batch['obs1'])
                    logger.store(**results)
                    logger.store(Alpha=alpha.numpy())

        # End of epoch wrap-up.
        if ((t + 1) % log_every == 0) or (t + 1 == total_steps):
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch.
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TaskRet', average_only=True)
            logger.log_tabular('TestTaskRet', average_only=True)
            logger.log_tabular('TaskSolved', average_only=True)
            logger.log_tabular('TestTaskSolved', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            for idx in range(ac_number):
                logger.log_tabular(f'Q1Vals_{idx + 1}', with_min_and_max=True)
                logger.log_tabular(f'Q2Vals_{idx + 1}', with_min_and_max=True)
                logger.log_tabular(f'QDiff_{idx + 1}', with_min_and_max=True)

            if autotune_alpha:
                logger.log_tabular('AlphaLoss')
                logger.log_tabular('ActorEntropy')
                logger.log_tabular('Alpha')
                logger.log_tabular('TargetEntropy', target_entropy)

            iter_end_time = time.time()
            steps_elapsed = (t+1) % log_every or log_every
            time_elapsed = iter_end_time - iter_begin_time
            logger.log_tabular('StepsPerSecond', steps_elapsed / time_elapsed)
            iter_begin_time = iter_end_time
            logger.log_tabular('Time', time.time() - start_time)

            logger.dump_tabular()

        # Save model.
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if save_path is not None:
                tf.keras.models.save_model(actor, save_path)
