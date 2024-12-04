from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.experimental import io_callback

from utils import Logger, Transition


@struct.dataclass
class NetworkState:
    """
    This class stores the state of a network in its associated optimizer
    """
    model_parameters: eqx.Module
    optimizer_state: optax.OptState


@struct.dataclass
class PolicyState:
    """
    Base class to store the state of a policy network. By default it is the actor network
    """
    actor_network_state: NetworkState


TPolicyState = TypeVar("TPolicyState", bound=PolicyState)
LogDict = dict[str, float]


class Network():
    """
    Helper class used to handle neural networks. It handles the building, storing, calling and updating of the network.
    You should interact with your policy network only through this class.
    """
    def __init__(self, model: eqx.Module, learning_rate: float = 1e-3, lr_decay: Optional[int] = None):
        """
        Creates and initializes an optimizer for the specified neural network.

        :param eqx.Module model: Neural network to use
        :param float learning_rate: Learning rate of the optimizer
        :param Optional[int] lr_decay: If provided, corresponds to the number of gradients steps to run before the learning is halved
        """
        model_state, self.model_functions = eqx.partition(model, eqx.is_array)
        if lr_decay is None:
            learning_rate_scheduler = learning_rate
        else:
            learning_rate_scheduler = optax.schedules.exponential_decay(learning_rate, lr_decay, min(learning_rate, 1e-5), 0.5)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(learning_rate=learning_rate_scheduler)
        )
        self.init_state = NetworkState(model_state, self.optimizer.init(model_state))

    def get_init_state(self) -> NetworkState:
        """
        Returns the initial state of the network and its optimizer.
        """
        return self.init_state

    def get_logits(self, model_parameters: eqx.Module, model_input: jax.Array) -> jax.Array:
        """
        Computes the output of the model for a given **unbatched** input

        :param eqx.Module model_parameters: Parameters of the model to use (from NetworkState.model_parameters)
        :param jax.Array model_input: Single input of shape [4, 4, 31]

        :return logits (jax.Array): Output logits of shape [4] or [1] depending on whether we are using the actor or critic model
        """
        model = eqx.combine(model_parameters, self.model_functions)
        return model(model_input)

    def get_batch_logits(self, model_parameters: eqx.Module, model_inputs: jax.Array) -> jax.Array:
        """
        Computes the outputs of the model for a given batch of inputs

        :param eqx.Module model_parameters: Parameters of the model to use (from NetworkState.model_parameters)
        :param jax.Array model_input: Batch of inputs of shape [batch_size, 4, 4, 31]

        :return logits (jax.Array): Output logits of shape [batch_size, 4] or [batch_size, 1] depending on whether we are using the actor or critic model
        """
        return jax.vmap(self.get_logits, in_axes=(None, 0))(model_parameters, model_inputs)

    def update(self, state: NetworkState, gradients: eqx.Module) -> NetworkState:
        """
        Updates the model parameters and optimizer state based on the loss gradients.

        :param NetworkState state: State of the network and optimizer to be updated
        :param eqx.Module gradients: Gradients of the loss function with respect to each parameter of the model

        :return updated_state (NetworkState): Updated state of the network and optimizer
        """
        updates, updated_optimizer_state = self.optimizer.update(gradients, state.optimizer_state)
        updated_model_parameters = eqx.apply_updates(state.model_parameters, updates)
        return NetworkState(updated_model_parameters, updated_optimizer_state)


class Policy(ABC, Generic[TPolicyState]):
    """
    Base class used to define a policy. This class implements all the functions needed to compute action probabilities for given states.
    """
    logger_entries: list[str]

    @abstractmethod
    def get_init_state(self) -> TPolicyState:
        """
        Returns the initial state of the policy to be used by the trainer.
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, model_parameters: eqx.Module, observation: jax.Array, action_mask: jax.Array) -> jax.Array:
        """
        Computes the probabilities of taking any action in given **single** state.

        :param eqx.Module model_parameters: Parameters of the policy network used to compute action probabilities (actor network)
        :param jax.Array observation: Unbatched observation of shape [4, 4, 31] of the state for which to compute action probabilities
        :param jax.Array action_mask: Mask of shape [4] of the valid actions to take in given state (1 = valid, 0 = forbidden)

        :return action_probabilities (jax.Array): Array of shape [4] of the probabilities of taking any action in given state.
            Forbidden actions must have a probability of 0.
        """
        pass

    def sample_action(self, key: chex.PRNGKey, policy_state: TPolicyState, observation: jax.Array, action_mask: jax.Array) -> jax.Array:
        """
        Samples one action from the current policy distribution and for a given state.

        :param chex.PRNGKey key: Random key
        :param PolicyState policy_state: Current state of the policy
        :param jax.Array observation: Unbatched observation of shape [4, 4, 31] of the state in which to sample an action
        :param jax.Array action_mask: Mask of shape [4] of the valid actions to take in given state (1 = valid, 0 = forbidden)

        :return sampled_action jax.Array: Sampled action, should be a jax.Array with empty shape and dtype of jnp.int32
        """
        ### ------------------------- To implement -------------------------
        action_probs = self.get_action_probabilities(policy_state.actor_network_state.model_parameters, observation, action_mask)
        action_probs = action_probs.astype(jnp.float32)
    
        # Apply action mask: set invalid actions to zero probability and normalize the remaining probabilities
        masked_probs = action_probs * action_mask
        masked_probs = masked_probs / masked_probs.sum()  # Normalize to sum to 1
        
        # Sample an action based on the probabilities
        sampled_action = jax.random.choice(key, jnp.arange(action_probs.shape[0]), p=masked_probs)
        
        return sampled_action.astype(jnp.int32)

        ### ----------------------------------------------------------------

    def actions_to_probabilities(self, model_parameters: eqx.Module, observations: jax.Array, actions: jax.Array, action_masks: jax.Array) -> jax.Array:
        """
        Computes the probabilities of taking each provided action in their corresponding state.

        :param eqx.Module model_parameters: Parameters of the actor network
        :param jax.Array observations: Batch of observations of shape [batch_size, 4, 4, 31] of the states in which the actions have been taken
        :param jax.Array actions: Batch of actions of shape [batch_size]
        :param jax.Array action_masks: Batch of valid action masks of shape [batch_size, 4] indicating which actions are valid in the corresponding states

        :return action_probabilities (jax.Array): Array of shape [batch_size] containing the probabilities of taking each action in the corresponding states
        """
        all_probabilities = jax.vmap(self.get_action_probabilities, in_axes=(None, 0, 0))(model_parameters, observations, action_masks)
        return jnp.take_along_axis(all_probabilities, actions[:, None], axis=1).squeeze(1)

    @abstractmethod
    def compute_loss(self, model_parameters: eqx.Module, transitions: Transition) -> tuple[float, LogDict]:
        """
        Computes the policy optimization objective to be differentiated with respect to the model parameters.

        :param eqx.Module model_parameters: Current parameters of the model
        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension)

        :return loss (float): Value of the loss
        :return log_dict (dict[str, float]): Dictionnary containing entries to log
        """
        pass

    @abstractmethod
    def update(self, state: TPolicyState, transitions: Transition) -> TPolicyState:
        """
        Updates the state of the policy by performing gradient descent/ascent on its model parameters on a batch of transitions.

        :param PolicyState state: Current state of the policy
        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension)

        :return updated_state (PolicyState): Updated policy state
        """
        pass

    def set_logger(self, logger: Logger) -> None:
        """
        Sets the logger to be used to record values.
        """
        self.logger = logger


class RandomPolicy(Policy[PolicyState]):
    logger_entries = []

    def __init__(self):
        self.network = Network(eqx.nn.Identity())

    def get_init_state(self):
        return PolicyState(self.network.get_init_state())

    def get_action_probabilities(self, model_parameters: eqx.Module, observation: jax.Array, action_mask: jax.Array) -> jax.Array:
        return action_mask.astype(jnp.float32) / action_mask.astype(jnp.float32).sum()

    def compute_loss(self, model_parameters: eqx.Module, transitions: Transition) -> tuple[float, LogDict]:
        return jnp.array(0, dtype=jnp.float32), {}

    def update(self, state: PolicyState, transitions: Transition) -> PolicyState:
        return state


@struct.dataclass
class ReinforcePolicyState(PolicyState):
    """
    State of the REINFORCE policy, containing the state of the actor network.
    """
    pass


class ReinforcePolicy(Policy[ReinforcePolicyState]):
    """Class reprenting the REINFORCE policy"""
    logger_entries = ["Actor loss"]

    def __init__(self, actor: Network, discount_factor: float):
        """
        Creates a REINFORCE policy with an actor network and a discount factor.

        :param Network actor: Actor network to be used by the policy
        :param float discount_factor: Discount factor to be used by the policy
        """
        self.actor = actor
        self.discount_factor = discount_factor

    def get_init_state(self) -> ReinforcePolicyState:
        return ReinforcePolicyState(self.actor.get_init_state())

    @staticmethod   # This function doesn't depend on the state of the policy
    def compute_discounted_returns(transitions: Transition, discount_factor: float) -> jax.Array:
        """
        Returns the discounted return in each visited state of the provided trajectories

        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension).
            All transitions follow each other and the first transition of the batch is the reset state of the environment.
            However, the batch may contain multiple episodes, meaning several transitions may verify `done == True`, which must be handled
            accordingly.
        :param float discount_factor: Discount factor to use
        """
        ### ------------------------- To implement -------------------------

        rewards = transitions.reward  
        rewards = jnp.clip(rewards, -20.0, 20.0) #Clip rewards
        dones = transitions.done      
        time_steps = dones.shape[0]
        
        returns = jnp.zeros_like(rewards)
                
        # Reverse time order: we will use `jax.lax.scan`
        def step(carry, t):
            returns_next = carry  # Discounted return from the next time step
            reward_t = rewards[t]
            done_t = dones[t]

            # Compute discounted return for time step t
            returns_t = reward_t + discount_factor * returns_next * (1.0 - done_t)
            return returns_t, returns_t

        returns, _ = jax.lax.scan(
            step,
            returns,               # Initial carry (final returns start as 0)
            jnp.arange(time_steps - 1, -1, -1)   # Time steps in reverse order)
        )
                                              
        return returns[::-1]

    def get_action_probabilities(self, model_parameters: eqx.Module, observation: jax.Array, action_mask: jax.Array) -> jax.Array:
        """
        Computes the probabilities of taking any action in given **single** state.

        :param eqx.Module model_parameters: Parameters of the policy network used to compute action probabilities (actor network)
        :param jax.Array observation: Unbatched observation of shape [4, 4, 31] of the state for which to compute action probabilities
        :param jax.Array action_mask: Mask of shape [4] of the valid actions to take in given state (1 = valid, 0 = forbidden)

        :return action_probabilities (jax.Array): Array of shape [4] of the probabilities of taking any action in given state.
            Forbidden actions must have a probability of 0.
        """
        ### ------------------------- To implement -------------------------

        logits = self.actor.get_logits(model_parameters, observation)

        exp_h = jnp.exp(logits) 
        action_probabilities = exp_h / exp_h.sum(axis=0)

        action_probabilities = action_probabilities * action_mask
        action_probabilities = action_probabilities/action_probabilities.sum(axis=0)

        return action_probabilities

        ### ----------------------------------------------------------------

    def compute_loss(self, model_parameters: eqx.Module, transitions: Transition) -> tuple[float, LogDict]:
        """
        Computes the policy optimization objective to be differentiated with respect to the model parameters.

        :param eqx.Module model_parameters: Current parameters of the model
        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension)

        :return loss (float): Value of the loss
        :return log_dict (dict[str, float]): Dictionnary containing entries to log
        """
        ### ------------------------- To implement -------------------------
        observations, actions, action_mask = (transitions.observation,transitions.action,transitions.action_mask)

        discounted_returns = self.compute_discounted_returns(transitions, self.discount_factor)

        def calc_log_prob(carry, idx):
            action_probabilities = self.get_action_probabilities(
                model_parameters, observations[idx], action_mask[idx]
            )
            log_probabilities = jnp.log(action_probabilities[actions[idx]])
            return carry, log_probabilities

        _, action_log_probabilities = jax.lax.scan(
            calc_log_prob,
            None,  # carry is not needed
            jnp.arange(observations.shape[0]),
        )

        loss = -jnp.mean(discounted_returns*action_log_probabilities)
        ### ----------------------------------------------------------------
        return loss, {"Actor loss": loss}

    def update(self, state: ReinforcePolicyState, transitions: Transition) -> ReinforcePolicyState:
        (loss, loss_dict), gradients = jax.value_and_grad(self.compute_loss, has_aux=True)(
            state.actor_network_state.model_parameters,
            transitions
        )

        new_actor_state = self.actor.update(state.actor_network_state, gradients)
        io_callback(self.logger.record, None, loss_dict)
        return ReinforcePolicyState(new_actor_state)


@struct.dataclass
class ActorCriticState(PolicyState):
    """
    State of the ActorCritic and REINFORCE with baseline policies. It contains the state of both the actor and critic (or value) networks.
    """
    critic_network_state: NetworkState

TActorCriticState = TypeVar("TActorCriticState", bound=ActorCriticState)


class BaseActorCriticPolicy(Policy[TActorCriticState], Generic[TActorCriticState]):
    """Base class for REINFORCE with baseline and Actor-Critic policies"""
    logger_entries = ["Actor loss", "Critic loss"]

    def __init__(self, actor: Network, critic: Network, discount_factor: float):
        """
        Creates an Actor-Critic-based policy with an actor network, a critic/value network and a discount factor.

        :param Network actor: Actor network to be used by the policy
        :param Network critic: Critic network (or value network) to be used by the policy
        :param float discount_factor: Discount factor to be used by the policy
        """
        self.actor = actor
        self.critic = critic
        self.discount_factor = discount_factor

    def get_action_probabilities(self, model_parameters: eqx.Module, observation: jax.Array, action_mask: jax.Array) -> jax.Array:
        """
        Computes the probabilities of taking any action in given **single** state.

        :param eqx.Module model_parameters: Parameters of the policy network used to compute action probabilities (actor network)
        :param jax.Array observation: Unbatched observation of shape [4, 4, 31] of the state for which to compute action probabilities
        :param jax.Array action_mask: Mask of shape [4] of the valid actions to take in given state (1 = valid, 0 = forbidden)

        :return action_probabilities (jax.Array): Array of shape [4] of the probabilities of taking any action in given state.
            Forbidden actions must have a probability of 0.
        """
        ### ------------------------- To implement -------------------------
        logits = self.actor.get_logits(model_parameters,observation)  # Shape [4], raw logits for each action

        exp_h = jnp.exp(logits) 
        action_probabilities = exp_h / exp_h.sum(axis=0)

        action_probabilities = action_probabilities * action_mask
        action_probabilities = action_probabilities/action_probabilities.sum(axis=0)

        return action_probabilities

        ### ----------------------------------------------------------------


class ActorCriticPolicy(BaseActorCriticPolicy[ActorCriticState]):
    """Class representing the Actor-Critic policy"""
    def get_init_state(self) -> ActorCriticState:
        return ActorCriticState(self.actor.get_init_state(), self.critic.get_init_state())

    def compute_loss(self, model_parameters: tuple[eqx.Module, eqx.Module], transitions: Transition) -> tuple[float, dict[str, float]]:
        """
        Computes the policy optimization objective to be differentiated with respect to the model parameters from the actor and critic networks.

        :param tuple[eqx.Module, eqx.Module] model_parameters: Tuple of the current parameters of the model in the following order:
            (actor_parameters, critic_parameters)
        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension)

        :return loss (float): Value of the loss
        :return log_dict (dict[str, float]): Dictionnary containing entries to log
        """
        actor_parameters, critic_parameters = model_parameters
        ### ------------------------- To implement -------------------------
        observations, actions, action_mask, rewards, dones, next_observation = (transitions.observation, transitions.action, transitions.action_mask, transitions.reward,
                                                       transitions.done, transitions.next_observation)

        predicted_values = self.critic.get_batch_logits(critic_parameters, observations)  
        next_values = self.critic.get_batch_logits(critic_parameters, next_observation)  

        rewards = jnp.clip(rewards, -20, 20)

        td_error = rewards + self.discount_factor*next_values*(1.0 - dones) - predicted_values

        td_objective = predicted_values - jax.lax.stop_gradient(rewards + self.discount_factor*next_values*(1.0 - dones))

        def compute_log_probability(carry, idx):
            action_probabilities = self.get_action_probabilities(
                actor_parameters, observations[idx], action_mask[idx]
            )
            log_probabilities = jnp.log(action_probabilities[actions[idx]])
            return carry, log_probabilities

        _, action_log_probabilities = jax.lax.scan(
            compute_log_probability,
            None,  # carry is not needed
            jnp.arange(observations.shape[0]),
        )
        
        #Compute losses
        actor_loss =  -jnp.mean(action_log_probabilities*jax.lax.stop_gradient(td_error))
        critic_loss = 0.5*jnp.mean(td_objective**2) 
        loss = actor_loss + critic_loss
        ### ----------------------------------------------------------------

        loss_dict = {
            "Actor loss": actor_loss,
            "Critic loss": critic_loss
        }
        return loss, loss_dict

    def update(self, state: ActorCriticState, transitions: Transition) -> ActorCriticState:
        model_parameters = (state.actor_network_state.model_parameters, state.critic_network_state.model_parameters)
        (loss, loss_dict), (actor_gradients, critic_gradients) = jax.value_and_grad(self.compute_loss, has_aux=True)(
            model_parameters,
            transitions
        )

        new_actor_state = self.actor.update(state.actor_network_state, actor_gradients)
        new_critic_state = self.critic.update(state.critic_network_state, critic_gradients)
        io_callback(self.logger.record, None, loss_dict)
        return ActorCriticState(new_actor_state, new_critic_state)


class ReinforceBaselinePolicy(ActorCriticPolicy, ReinforcePolicy):
    """Class representing the REINFORCE with baseline policy"""
    logger_entries = ["Actor loss", "Value network loss"]

    def compute_loss(self, model_parameters: tuple[eqx.Module, eqx.Module], transitions: Transition) -> tuple[float, dict[str, float]]:
        """
        Computes the policy optimization objective to be differentiated with respect to the model parameters from the actor and critic networks.

        :param tuple[eqx.Module, eqx.Module] model_parameters: Tuple of the current parameters of the model in the following order:
            (actor_parameters, critic_parameters)
        :param Transition transitions: Batch of transitions (each attribute of transitions is an array with a batch dimension)

        :return loss (float): Value of the loss
        :return log_dict (dict[str, float]): Dictionnary containing entries to log
        """
        actor_parameters, critic_parameters = model_parameters

        observations, actions, action_mask = (transitions.observation, transitions.action, transitions.action_mask)

        returns = self.compute_discounted_returns(transitions, discount_factor=self.discount_factor)

        crt_logits = self.critic.get_batch_logits(critic_parameters, observations)

        advantage = returns - jax.lax.stop_gradient(crt_logits)

        def compute_log_prob(carry, idx):
            action_probabilities = self.get_action_probabilities(
                actor_parameters, observations[idx], action_mask[idx]
            )
            log_probabilities = jnp.log(action_probabilities[actions[idx]])
            return carry, log_probabilities

        _, action_log_probabilities = jax.lax.scan(
            compute_log_prob,
            None,  # carry is not needed
            jnp.arange(observations.shape[0]),
        )
        
        actor_loss =  -jnp.mean(action_log_probabilities*advantage)
        critic_loss = jnp.mean(advantage**2)   
        loss = actor_loss + critic_loss
        ### ----------------------------------------------------------------

        loss_dict = {
            "Actor loss": actor_loss,
            "Value network loss": critic_loss
        }
        return loss, loss_dict
