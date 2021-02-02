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


def assert_callable(name, value):
    if not callable(value):
        raise TypeError(
            f"Expected argument {name} to be callable, instead it is "
            f"{type(value).__name__}"
        )


def assert_iterable(name, value):
    try:
        iter(value)
    except TypeError:
        raise TypeError(
            f"Expected argument {name} to be iterable, instead it is "
            f"{type(value).__name__}."
        )


def assert_positive(name, value):
    if value <= 0:
        raise ValueError(f"Expected argument {name} to be positive, but it is {value}.")


def assert_non_negative(name, value):
    if value < 0:
        raise ValueError(
            f"Expected argument {name} to be non-negative, but it is {value}."
        )


def assert_all_int(name, value):
    for i, element in enumerate(value):
        try:
            assert_type(name, value, int)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain ints, but found a "
                f"{type(element).__name__} at index {i}."
            )


def assert_all_callable(name, value):
    for i, element in enumerate(value):
        try:
            assert_all_callable(name, value)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain callables, but found a "
                f"{type(element).__name__} at index {i}."
            )


def assert_all_non_negative(name, value):
    for i, element in enumerate(value):
        try:
            assert_non_negative(name, value)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain non-negative elements, but "
                f"found {element} at index {i}."
            )
