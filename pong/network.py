from typing import Any
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class PongRL(nn.Module):
    hidden_dim = 200
    def __init__(self, width:int, height:int):
        super().__init__()
        W, H =  width, height
        self.fc1 = nn.Linear(input_dims = W*H, output_dims=self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, 2, bias=False)

    def __call__(self, Xin: mx.array):
        x = nn.relu(self.fc1(Xin))
        logits = self.fc2(x)
        return logits
    
    def decide(self, logits: mx.array)->bool:
        out = mx.random.categorical(logits)
        probs = mx.softmax(logits)
        # print(probs, out, probs[out])
        logprob = probs[0,out].log()
        # print(logprob)
        print(bool(out),probs[0,out])
        return bool(out), logprob
    
if __name__ == '__main__':
    model = PongRL(width=210, height = 160)
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"{num_params=}")
    xin = mx.random.normal((10,210,160))
    xin = mx.flatten(xin, start_axis=-2,end_axis=-1)
    a = model(xin)
    print(a.shape)