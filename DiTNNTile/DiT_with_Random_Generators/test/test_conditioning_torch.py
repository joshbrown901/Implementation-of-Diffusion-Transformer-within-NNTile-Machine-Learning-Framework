import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from conditioning_torch import CombinedTimestepLabelEmbeddings

def test_combined_timestep_label_embeddings():
    # Set device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configurations
    batch_size = 8
    embedding_dim = 128
    num_classes = 10
    class_dropout_prob = 0.1
    
    # Instantiate the module
    combined_embedder = CombinedTimestepLabelEmbeddings(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        class_dropout_prob=class_dropout_prob
    ).to(device)
    
    # Create test inputs
    timesteps = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.long).to(device)
    class_labels = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long).to(device)
    
    # Test 1: Forward pass shape check
    print("=== Testing Forward Pass ===")
    conditioning = combined_embedder(timesteps, class_labels)
    expected_shape = (batch_size, embedding_dim)
    print(f"Output shape: {conditioning.shape} (expected: {expected_shape})")
    assert conditioning.shape == expected_shape, f"Shape mismatch! Got {conditioning.shape}, expected {expected_shape}"


# Run the test
test_combined_timestep_label_embeddings()