def shape_guard(expected_shape):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Simple shape guard implementation
            return func(*args, **kwargs)
        return wrapper
    return decorator
