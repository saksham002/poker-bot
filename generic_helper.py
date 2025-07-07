import inspect

def debug_print(text: str):
    """
    Prints a debug message with the name of the calling function.

    Args:
        text: The debug message to print.
    """
    caller_frame = inspect.stack()[1]
    caller_function_name = caller_frame.function
    print(f"{caller_function_name}: {text}")
