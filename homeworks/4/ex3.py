from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import data
from data import enc_data

# This exercise relies on solutions from ex1 and ex2
from ex1 import create_model
from ex2 import calculate_loss

# For reproducibility (optional)
torch.manual_seed(0)

# Create the language prediction model
model = create_model(
    alpha_size=None,    # TODO: provide appropriate alphabet size
    emb_size=None,      # TODO: choose appropriate embedding size
    class_num=None      # TODO: provide the number of output classes
)

# TODO: Create optimizer (e.g. torch.optim.Adam) with appropriate arguments
optim: torch.optim.Optimizer = None

# Perform SGD for a selected number of epochs
epoch_num = None        # TODO: select appropriate number of epochs
for k in range(epoch_num):
    for x, y in enc_data:
        # TODO: Calculate the loss, call backward
        # TODO: Apply the optimisation step
        pass

#################################################
# EVALUATION SECTION START: DO NOT MODIFY!
#################################################

# Let's verify the final losses
total_loss = sum(
    calculate_loss(model(x), y).item()
    for x, y in enc_data
)

# Evaluation: total loss should be smaller than 1
if total_loss < 1.0:
    print(f"OK: final total loss {total_loss} < 1")
else:
    print(f"FAILED: final total loss {total_loss} >= 1")

#################################################
# EVALUATION SECTION END
#################################################