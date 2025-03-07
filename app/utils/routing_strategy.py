from enum import Enum

class RoutingStrategy(str, Enum):
    """Strategy used to select an API instance."""
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin" 
    WEIGHTED = "weighted"
    LEAST_LOADED = "least_loaded"
    FAILOVER = "failover"
    MODEL_SPECIFIC = "model_specific"
