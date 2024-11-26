import jax
import pickle
from pathlib import Path

from agents import Network, ReinforceBaselinePolicy
from models import CNN_2048
from trainers import ReinforceTrainer

SEED = 42
NUM_EPISODES = 1_500
SAVE_ROOT = Path("results")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(SEED)
rng, actor_key, critic_key, baseline_key = jax.random.split(rng, 4)
actor_model = CNN_2048(actor_key, 4)
critic_model = CNN_2048(critic_key, 1)

actor = Network(actor_model, learning_rate=1e-3)
critic = Network(critic_model, learning_rate=1e-3)
policy = ReinforceBaselinePolicy(actor, critic, discount_factor=0.99)

trainer = ReinforceTrainer(rng, policy)

baseline_perfs = trainer.get_random_policy_performances(baseline_key, is_reinforce_trainer=True)
final_trainer_state = trainer.train(NUM_EPISODES)
trainer.plot_results(baseline=baseline_perfs)

figure_save_path = SAVE_ROOT / "figures/reinforce_baseline"
figure_save_path.mkdir(parents=True, exist_ok=True)
trainer.save_figures(figure_save_path, baseline=baseline_perfs)

result_save_path = SAVE_ROOT / "reinforce_baseline_logger_history.pickle"
with result_save_path.open("wb") as f:
    pickle.dump(
        {
            "history": trainer.logger.history,
            "episodes": trainer.logger.episodes
        },
        f
    )
