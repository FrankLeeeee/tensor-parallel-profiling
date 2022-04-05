import colossalai
from colossalai.nn.layer.utils import CheckpointModule

class LinearModel(CheckpointModule):

    def __init__(self, dim, mlp_ratio, checkpoint=False):
        super().__init__(checkpoint)
        self.dense_1 = colossalai.nn.Linear(dim, dim*mlp_ratio)
        self.dense_2 = colossalai.nn.Linear(dim*mlp_ratio, dim)

    def forward(self, x):
        return self.dense_2(self.dense_1(x))

class LayerNormModel(CheckpointModule):
    def __init__(self, dim, checkpoint=False):
        super().__init__(checkpoint)
        self.norm_1 = colossalai.nn.LayerNorm(dim)
        self.norm_2 = colossalai.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm_2(self.norm_1(x))

