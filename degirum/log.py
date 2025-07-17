#
# _log.py - DeGirum Python SDK: log functionality
# Copyright DeGirum Corp. 2022
#
# Implements common DeGirum logging functionality
#

import logging
import functools
import time
import sys

from typing import Callable, TypeVar, Optional, Union, overload, Awaitable

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec


logger = logging.getLogger(__name__)


class DGLog:
    """Console logging class with programmable verbosity."""

    def __dir__(self):
        return ["set_verbose_state", "print"]

    _prefix = ""
    _suppress = False

    @staticmethod
    def set_verbose_state(state: bool):
        """Set log verbosity state.

        Args:
            state: If `True`, then log prints messages to console, otherwise no messages printed.
        """
        DGLog._suppress = not state

    @staticmethod
    def print(message: str):
        """Print message to log according to current verbosity level.

        Args:
            message: Message string to print.
        """
        if not DGLog._suppress:
            print(DGLog._prefix + message)


R = TypeVar("R")
P = ParamSpec("P")


@overload
def log_wrap(  # noqa: E704
    f: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def log_wrap(  # noqa: E704
    *,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def log_wrap(
    f: Optional[Callable[P, R]] = None,
    *,
    log_level: int = logging.DEBUG,
) -> Union[
    Callable[P, R],
    Callable[[Callable[P, R]], Callable[P, R]],
]:
    """Decorator to log function entry and exit with execution time.

    Args:
        f (Callable): Sync function to log
        log_level: Logging level of the log entries.
    """
    if f is None:
        return functools.partial(log_wrap, log_level=log_level)

    @functools.wraps(f)
    def sync_wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        t1 = time.time_ns()
        try:
            logger.log(log_level, f"/ {f.__qualname__}")
            return f(*args, **kwargs)
        finally:
            t2 = time.time_ns()
            logger.log(log_level, f"\\ {f.__qualname__} {(t2 - t1) * 1e-3}us ")

    return sync_wrap


@overload
def async_log_wrap(  # noqa: E704
    f: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]: ...


@overload
def async_log_wrap(  # noqa: E704
    *,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def async_log_wrap(
    f: Optional[Callable[P, Awaitable[R]]] = None,
    *,
    log_level: int = logging.DEBUG,
) -> Union[
    Callable[P, Awaitable[R]],
    Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]],
]:
    """Decorator to log async function entry and exit with execution time.

    Args:
        f (Callable): Async function to log.
        log_level: Logging level of the log entries.
    """
    if f is None:
        return functools.partial(async_log_wrap, log_level=log_level)

    @functools.wraps(f)
    async def async_wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        t1 = time.time_ns()
        try:
            logger.log(log_level, f"/ {f.__qualname__}")
            return await f(*args, **kwargs)
        finally:
            t2 = time.time_ns()
            logger.log(log_level, f"\\ {f.__qualname__} {(t2 - t1) * 1e-3}us ")

    return async_wrap
