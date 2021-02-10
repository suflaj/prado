from typing import Tuple, Union


def assert_type(name: str, value, expected_type):
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


def assert_not_empty(name: str, value):
    if len(value) == 0:
        raise ValueError(f"Expected argument {name} to not be empty.")


def assert_non_negative(name, value):
    if value < 0:
        raise ValueError(
            f"Expected argument {name} to be non-negative, but it is {value}."
        )


def assert_in_range(
    name: str,
    value,
    value_range: Tuple[Union[float, int], Union[float, int]],
    inclusivity: Tuple[bool, bool] = (True, False),
):
    s = ["or equal to" if inclusivity[i] else "than" for i in range(2)]

    if value < value_range[0] or (value == value_range[0] and not inclusivity[0]):
        raise ValueError(
            f"Expected argument {name} to be less {s[0]} {value_range[0]}, instead it "
            f"is {value}."
        )
    elif value > value_range[1] or (value == value_range[1] and not inclusivity[1]):
        raise ValueError(
            f"Expected argument {name} to be greater {s[1]} {value_range[1]}, "
            f"instead it is {value}."
        )


def assert_all_int(name, value):
    for i, element in enumerate(value):
        try:
            assert_type(name, element, int)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain ints, but found a "
                f"{type(element).__name__} at index {i}."
            )


def assert_all_float(name, value):
    for i, element in enumerate(value):
        try:
            assert_type(name, element, float)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain floats, but found a "
                f"{type(element).__name__} at index {i}."
            )


def assert_all_callable(name, value):
    for i, element in enumerate(value):
        try:
            assert_callable(name, element)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain callables, but found a "
                f"{type(element).__name__} at index {i}."
            )


def assert_all_non_negative(name, value):
    for i, element in enumerate(value):
        try:
            assert_non_negative(name, element)
        except ValueError:
            raise ValueError(
                f"Expected argument {name} to only contain non-negative elements, but "
                f"found {element} at index {i}."
            )
