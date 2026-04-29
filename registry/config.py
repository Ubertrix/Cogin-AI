class SystemConfig:
    """Global Project Settings (Hyperparameters)"""
    PROJECT_NAME = "Cogni Pro"
    COMPANY_NAME = "Ubertrix"
    VERSION = "1.0.0-COGNI"
    EMBEDDING_DIM = 1024       # Dimension for word embeddings (Neural Space Expansion)
    MAX_EXPERTS = 8            # Maximum number of allowed experts
    LEARNING_RATE = 0.001      # Learning rate
    STABILITY_THRESHOLD = 0.85 # Stability threshold for creating new cells
