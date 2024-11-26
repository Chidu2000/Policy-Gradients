import jax
import pickle
from pathlib import Path

from agents import Network, ActorCriticPolicy
from models import CNN_2048
from trainers import ActorCriticTrainer

SEED = 42
NUM_EPISODES = 1_500
BATCH_SIZE = 200  # Modify the batch size here
SAVE_ROOT = Path("results")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(SEED)
rng, actor_key, critic_key, baseline_key = jax.random.split(rng, 4)
actor_model = CNN_2048(rng, 4)
critic_model = CNN_2048(rng, 1)

actor = Network(actor_model, learning_rate=1e-3)
critic = Network(critic_model, learning_rate=1e-3)
policy = ActorCriticPolicy(actor, critic, discount_factor=0.99)

trainer = ActorCriticTrainer(rng, policy, batch_size=BATCH_SIZE)

baseline_perfs = trainer.get_random_policy_performances(baseline_key, is_reinforce_trainer=False)
final_trainer_state = trainer.train(NUM_EPISODES)
trainer.plot_results(baseline=baseline_perfs)

figure_save_path = SAVE_ROOT / f"figures/actor_critic_{BATCH_SIZE}"
figure_save_path.mkdir(parents=True, exist_ok=True)
trainer.save_figures(figure_save_path, baseline=baseline_perfs, suffix=str(BATCH_SIZE))

result_save_path = SAVE_ROOT / f"actor_critic_logger_history_{BATCH_SIZE}.pickle"
with result_save_path.open("wb") as f:
    pickle.dump(
        {
            "history": trainer.logger.history,
            "episodes": trainer.logger.episodes
        },
        f
    )
