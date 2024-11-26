import jax
import pickle
from pathlib import Path

from agents import Network, ReinforcePolicy
from models import CNN_2048
from trainers import ReinforceTrainer

SEED = 42
NUM_EPISODES = 1_500
SAVE_ROOT = Path("results")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(SEED)
rng, actor_key, baseline_key = jax.random.split(rng, 3)

actor_model = CNN_2048(actor_key, 4)
actor = Network(actor_model, learning_rate=1e-3)
policy = ReinforcePolicy(actor, discount_factor=0.99)

trainer = ReinforceTrainer(rng, policy)
baseline_perfs = trainer.get_random_policy_performances(baseline_key, is_reinforce_trainer=True)
final_trainer_state = trainer.train(NUM_EPISODES)
trainer.plot_results(baseline=baseline_perfs)

figure_save_path = SAVE_ROOT / "figures/reinforce"
figure_save_path.mkdir(parents=True, exist_ok=True)
trainer.save_figures(figure_save_path, baseline=baseline_perfs)

result_save_path = SAVE_ROOT / "reinforce_logger_history.pickle"
with result_save_path.open("wb") as f:
    pickle.dump(
        {
            "history": trainer.logger.history,
            "episodes": trainer.logger.episodes
        },
        f
    )
