"""Repository-local Isaac Lab tasks for G008."""


def register_tasks() -> None:
    from .registry import register_tasks as _register_tasks

    _register_tasks()


__all__ = ["register_tasks"]
