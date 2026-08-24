"""Repository-local Isaac Lab tasks for G006."""


def register_tasks() -> None:
    """Import Gym only when the simulator bootstrap explicitly registers tasks."""
    from .registry import register_tasks as _register_tasks

    _register_tasks()

__all__ = ["register_tasks"]
