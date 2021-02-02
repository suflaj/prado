def assert_type(name, value, expected_type):
    try:
        list(expected_type)
        type_name = (
            ", ".join([x.__name__ for x in expected_type[:-1]])
            + " or "
            + expected_type[-1].__name__
        )
    except TypeError:
        type_name = expected_type.__name__

    if not isinstance(value, expected_type):
        raise TypeError(
            f"Expected argument {name} to be of type {type_name}, instead it is "
            f"{type(value).__name__}."
        )


def assert_positive(name, value):
    if value <= 0:
        raise ValueError(f"Expected argument {name} to be positive, but it is {value}.")
