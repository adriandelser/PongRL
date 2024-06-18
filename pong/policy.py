import mlx.core as mx
import mlx.nn as nn
import mlx
import mlx.optimizers
from mlx.utils import tree_flatten

    
class Policy(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activations: str = [nn.relu],
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.activations = activations
        assert (
            len(self.layers) == len(self.activations) + 1
        ), "Number of layers and activations must match"

    def __call__(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x
    
    def decide(self, obs):
        logits = self(obs)
        action = mx.random.categorical(logits)
        return action.item()
    
class REINFORCE:
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        self.loss_and_grad_fn = nn.value_and_grad(self.policy, self.loss_fn)

    def get_action(self, obs):
        logits = self.policy(obs)
        action = mx.random.categorical(logits)
        return action.item()

    def loss_fn(self, observations, actions, rewards):
        log_probs = self.get_log_probs(observations, actions)
        loss = mx.sum(-log_probs * rewards)
        return loss

    def get_log_probs(self, observations, actions):
        logits = self.policy(observations)
        probs = nn.softmax(logits)[mx.arange(actions.shape[0]), actions]
        log_probs = mx.log(probs)
        return log_probs

    def update(self, observations, actions, rewards):
        loss, grads = self.loss_and_grad_fn(observations, actions, rewards)
        self.optimizer.update(self.policy, grads)
        mx.eval(self.policy.parameters(), self.optimizer.state)

def compute_discounted_rewards(rewards, gamma:float = 0.99):
    #ensure rewards array is of float dtype
    discounted_rewards = mx.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


class RolloutBuffer:
    def __init__(self):
        self.buffer = {}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(mx.array(value))

    def clear(self):
        self.buffer = {}

    def get(self, key):
        return mx.array(self.buffer.get(key, None))

    def __getitem__(self, key):
        return self.get(key)

    def __str__(self):
        return str(self.buffer)



if __name__ == '__main__':
    lr = 1e-3
    gamma = 0.99
    model = Policy(num_layers=1, input_dim=210*160, hidden_dim=200, output_dim=2)
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"{num_params=}")
    xin = mx.random.normal((10,210,160))
    xin = mx.flatten(xin, start_axis=-2,end_axis=-1)
    a = model(xin)
    print(a.shape, a)
    b = REINFORCE(model, optimizer=mlx.optimizers.AdamW(learning_rate = lr))
    logprobs = b.get_log_probs(xin, actions = mx.array([1,1,1,1,1,1,1,1,1,1]))
    print(logprobs)
    print(mx.arange(10))