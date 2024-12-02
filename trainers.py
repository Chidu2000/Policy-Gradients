from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Union, TypeVar
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pgx
import equinox as eqx
from flax import struct

from agents import ActorCriticState, Policy, PolicyState, RandomPolicy, ReinforcePolicyState
from utils import Logger, Transition
from jax.experimental import io_callback


TPolicyState = TypeVar("TPolicyState", bound=PolicyState)

@struct.dataclass
class TrainerState(Generic[TPolicyState]):
    current_env_state: pgx.State
    policy_state: TPolicyState
    rng: chex.PRNGKey
    episode: int                # jax.Array
    current_return: float       # jax.Array
    eval_counter: int           # jax.Array


class BaseTrainer(ABC, Generic[TPolicyState]):
    def __init__(
        self,
        rng: chex.PRNGKey,
        policy: Policy,
        evaluation_frequency: int = 50,
        n_evaluation_iterations: int = 5
    ) -> None:
        self.policy = policy
        self.evaluation_frequency = evaluation_frequency
        self.n_evaluation_iterations = n_evaluation_iterations
        self.env = pgx.make("2048")

        self.logger = Logger(self.policy.logger_entries + ["return", "evaluation return"])
        self.policy.set_logger(self.logger)

        rng, reset_key = jax.random.split(rng)
        self.env_state = self.env.init(reset_key)

        self.init_state = self.get_init_state(rng)

    def get_init_state(self, rng: chex.PRNGKey) -> TrainerState[TPolicyState]:
        return TrainerState(
            self.env_state,
            self.policy.get_init_state(),
            rng,
            jnp.array(0, dtype=jnp.uint32),
            jnp.array(0, dtype=jnp.float32),
            jnp.array(self.evaluation_frequency, dtype=jnp.int32)
        )

    @abstractmethod
    def train(self, num_episodes: int) -> TrainerState[TPolicyState]:
        pass

    def evaluate(self, rng: chex.PRNGKey, trainer_state: TrainerState, n_iter: int) -> None:
        def _compute_episode_return(init_trainer_state: TrainerState) -> float:
            def _step_env(val: tuple[TrainerState, bool]) -> tuple[TrainerState, bool]:
                trainer_state, _ = val
                rng, action_sample_key, step_key = jax.random.split(trainer_state.rng, 3)
                action = self.policy.sample_action(
                    action_sample_key,
                    trainer_state.policy_state,
                    trainer_state.current_env_state.observation,
                    trainer_state.current_env_state.legal_action_mask
                )
                next_env_state = self.env.step(trainer_state.current_env_state, action, step_key)
                done = next_env_state.terminated | next_env_state.truncated

                new_trainer_sate = TrainerState(
                    next_env_state,
                    trainer_state.policy_state,
                    rng,
                    trainer_state.episode + jnp.astype(done, jnp.uint32),
                    trainer_state.current_return + jnp.astype(~done, jnp.int32) * next_env_state.rewards[0],
                    trainer_state.eval_counter
                )

                return new_trainer_sate, done

            last_trainer_state, _ = jax.lax.while_loop(
                lambda val: ~val[1],
                _step_env,
                (init_trainer_state, jnp.array(0, dtype=jnp.bool))
            )

            return last_trainer_state.current_return

        keys = jax.random.split(rng, n_iter + 1)
        init_trainer_states = jax.vmap(TrainerState, in_axes=(None, None, 0, None, None, None))(
            self.env.init(keys[0]),
            trainer_state.policy_state,
            keys[1:],
            trainer_state.episode,
            jnp.array(0, dtype=jnp.float32),
            trainer_state.eval_counter
        )

        returns = jax.vmap(_compute_episode_return)(init_trainer_states)
        jax.debug.print(
            "Episode {epoch_num_episodes} : avg return = {avg_return}",
            epoch_num_episodes=trainer_state.episode,
            avg_return=jnp.mean(returns)
        )
        io_callback(self.logger.record, None, {"evaluation return": returns})

    def get_random_policy_performances(self, key: chex.PRNGKey, is_reinforce_trainer: bool = True) -> float:
        old_policy, old_logger = self.policy, self.logger
        if is_reinforce_trainer:
            old_max_step = self.max_steps_in_episode
        self.policy = RandomPolicy()
        self.logger = Logger(["evaluation return"])
        self.policy.set_logger(self.logger)
        self.max_steps_in_episode = 10_000

        reset_key, eval_key = jax.random.split(key)
        init_state = self.get_init_state(reset_key)
        self.evaluate(eval_key, init_state, 100)
        self.logger.log(0)
        random_policy_avg_return = np.mean(self.logger.history["evaluation return"])

        self.policy, self.logger = old_policy, old_logger
        if is_reinforce_trainer:
            self.max_steps_in_episode = old_max_step
        return random_policy_avg_return

    def plot_results(self, baseline: Optional[float] = None):
        baselines = {"return": baseline, "evaluation return": baseline} if baseline is not None else {}
        self.logger.plot("Training performances", baselines)

    def save_figures(self, save_root: Path, baseline: Optional[float] = None, suffix: str = ""):
        baselines = {"return": baseline, "evaluation return": baseline} if baseline is not None else {}
        self.logger.save_figures(save_root, baselines=baselines, suffix=suffix)


@struct.dataclass
class EpochCarry:
    trainer_state: TrainerState
    avg_return: float
    num_completed_episodes: int

class ReinforceTrainer(BaseTrainer[Union[ReinforcePolicyState, ActorCriticState]]):
    max_steps_in_episode = 500

    def train(self, num_episodes: int) -> TrainerState:
        def _train_epoch(trainer_state: TrainerState) -> TrainerState:
            def _step_env(carry: EpochCarry, step: int) -> tuple[EpochCarry, Transition]:
                rng, action_sample_key, step_key, reset_key = jax.random.split(carry.trainer_state.rng, 4)
                action = self.policy.sample_action(
                    action_sample_key,
                    carry.trainer_state.policy_state,
                    carry.trainer_state.current_env_state.observation,
                    carry.trainer_state.current_env_state.legal_action_mask
                )
                next_env_state = self.env.step(carry.trainer_state.current_env_state, action, step_key)
                done = next_env_state.terminated | next_env_state.truncated
                reward = next_env_state.rewards[0]
                next_env_state = jax.lax.cond(done, lambda: self.env.init(reset_key), lambda: next_env_state)

                transition = Transition(
                    carry.trainer_state.current_env_state.observation,
                    action,
                    next_env_state.observation,
                    reward,
                    done,
                    carry.trainer_state.current_env_state.legal_action_mask
                )

                new_trainer_state = TrainerState(
                    next_env_state,
                    carry.trainer_state.policy_state,
                    rng,
                    carry.trainer_state.episode,
                    jax.lax.cond(done, lambda: jnp.array(0, dtype=jnp.float32), lambda: carry.trainer_state.current_return + reward),
                    carry.trainer_state.eval_counter
                )

                new_carry = EpochCarry(
                    new_trainer_state,
                    jax.lax.cond(
                        done,
                        lambda: (carry.avg_return * carry.num_completed_episodes + carry.trainer_state.current_return + reward) / (carry.num_completed_episodes + 1),
                        lambda: carry.avg_return
                    ),
                    carry.num_completed_episodes + jnp.astype(done, jnp.int32)
                )
                return new_carry, transition


            rng, init_env_key, eval_key = jax.random.split(trainer_state.rng, 3)
            initial_trainer_state = TrainerState(
                self.env.init(init_env_key),
                trainer_state.policy_state,
                rng,
                trainer_state.episode,
                self.init_state.current_return,
                trainer_state.eval_counter
            )

            final_carry, transitions = jax.lax.scan(
                _step_env,
                EpochCarry(initial_trainer_state, jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)),
                jnp.arange(self.max_steps_in_episode)
            )

            epoch_avg_return, epoch_num_episodes = jax.lax.cond(
                final_carry.num_completed_episodes == 0,
                lambda: (final_carry.trainer_state.current_return, 1),
                lambda: (final_carry.avg_return, final_carry.num_completed_episodes)
            )

            new_policy_state = self.policy.update(final_carry.trainer_state.policy_state, transitions)
            new_trainer_state = eqx.tree_at(
                lambda t: (t.policy_state, t.episode, t.eval_counter),
                final_carry.trainer_state,
                replace=(
                    new_policy_state,
                    final_carry.trainer_state.episode + epoch_num_episodes.astype(jnp.uint32),
                    final_carry.trainer_state.eval_counter - epoch_num_episodes
                )
            )

            def _run_evaluation():
                self.evaluate(eval_key, new_trainer_state, self.n_evaluation_iterations)
                return new_trainer_state.eval_counter + self.evaluation_frequency

            new_eval_counter_value = jax.lax.cond(
                new_trainer_state.eval_counter <= 0,
                _run_evaluation,
                lambda: new_trainer_state.eval_counter
            )

            io_callback(self.logger.log, None, final_carry.trainer_state.episode + epoch_num_episodes, **{"return": epoch_avg_return})
            return eqx.tree_at(
                lambda t: t.eval_counter,
                new_trainer_state,
                replace=new_eval_counter_value
            )

        return jax.lax.while_loop(
            lambda trainer_state: trainer_state.episode < num_episodes,
            _train_epoch,
            self.init_state
        )


class ActorCriticTrainer(BaseTrainer[ActorCriticState]):
    def __init__(
        self,
        rng,
        policy,
        batch_size: int = 32,
        evaluation_frequency = 50,
        n_evaluation_iterations = 5
    ):
        super().__init__(rng, policy, evaluation_frequency, n_evaluation_iterations)
        self.batch_size = batch_size

    def train(self, num_episodes: int) -> TrainerState:
        def _train_episode(carry: EpochCarry) -> EpochCarry:
            def _step_env(carry: EpochCarry, i: int) -> tuple[EpochCarry, Transition]:
                trainer_state = carry.trainer_state
                rng, action_sample_key, step_key, reset_key = jax.random.split(trainer_state.rng, 4)
                action = self.policy.sample_action(
                    action_sample_key,
                    trainer_state.policy_state,
                    trainer_state.current_env_state.observation,
                    trainer_state.current_env_state.legal_action_mask
                )
                next_env_state = self.env.step(trainer_state.current_env_state, action, step_key)
                done = next_env_state.terminated | next_env_state.truncated
                reward = next_env_state.rewards[0]
                next_env_state = jax.lax.cond(done, lambda: self.env.init(reset_key), lambda: next_env_state)

                transition = Transition(
                    trainer_state.current_env_state.observation,
                    action,
                    next_env_state.observation,
                    reward,
                    done,
                    trainer_state.current_env_state.legal_action_mask
                )

                next_state = TrainerState(
                    next_env_state,
                    trainer_state.policy_state,
                    rng,
                    trainer_state.episode + jnp.astype(done, jnp.uint32),
                    jax.lax.cond(done, lambda: jnp.array(0, dtype=jnp.float32), lambda: trainer_state.current_return + reward),
                    trainer_state.eval_counter - jnp.astype(done, jnp.int32),

                )

                next_carry = EpochCarry(
                    next_state,
                    jax.lax.cond(
                        done,
                        lambda: trainer_state.current_return + reward,
                        lambda: carry.avg_return
                    ),
                    carry.num_completed_episodes + jnp.astype(done, jnp.int32)
                )

                return next_carry, transition

            init_carry = EpochCarry(
                carry.trainer_state,
                jax.lax.cond(carry.num_completed_episodes > 0, lambda: jnp.array(0, dtype=jnp.float32), lambda: carry.avg_return),
                jax.lax.cond(carry.num_completed_episodes > 0, lambda: jnp.array(0, dtype=jnp.int32), lambda: carry.num_completed_episodes)
            )

            last_carry, transitions = jax.lax.scan(
                _step_env,
                init_carry,
                jnp.arange(self.batch_size)
            )

            new_policy_state = self.policy.update(last_carry.trainer_state.policy_state, transitions)
            new_trainer_state = eqx.tree_at(
                lambda t: t.policy_state,
                last_carry.trainer_state,
                replace=new_policy_state
            )
            rng, eval_key = jax.random.split(new_trainer_state.rng)

            def _run_evaluation():
                self.evaluate(eval_key, new_trainer_state, self.n_evaluation_iterations)
                return new_trainer_state.eval_counter + self.evaluation_frequency

            new_eval_counter = jax.lax.cond(
                new_trainer_state.eval_counter <= 0,
                _run_evaluation,
                lambda: new_trainer_state.eval_counter
            )

            _ = jax.lax.cond(
                last_carry.num_completed_episodes > 0,
                lambda: io_callback(self.logger.log, None, new_trainer_state.episode.astype(jnp.int32), **{"return": last_carry.avg_return}),
                lambda: None
            )

            new_trainer_state = eqx.tree_at(
                lambda t: t.eval_counter,
                new_trainer_state,
                replace=new_eval_counter
            )

            return EpochCarry(new_trainer_state, last_carry.avg_return, last_carry.num_completed_episodes)

        return jax.lax.while_loop(
            lambda t: t.trainer_state.episode < num_episodes,
            _train_episode,
            EpochCarry(self.init_state, jnp.array(0, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))
        )
