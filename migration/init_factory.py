
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.instance.factory import InstanceFactory

# Initialize factory with new implementation
InstanceFactory.initialize(
    use_new_implementation=True,
    config_file_new=os.environ.get("INSTANCE_CONFIG_FILE", "instance_configs.json"),
    state_file_new=os.environ.get("INSTANCE_STATE_FILE", "instance_states.json")
)

print("Initialized InstanceFactory with new implementation")
