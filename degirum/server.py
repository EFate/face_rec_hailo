#
# server.py - DeGirum Python SDK: Degirum AI Server control
# Copyright DeGirum Corp. 2022
#
# Implements DeGirum AI Server start functionality
#


"""
DeGirum AI server launcher and model downloader.

!!! note

    The functionality of this module is now exposed via PySDK CLI.

The purpose of this module is to start DeGirum AI server:
```
python -m degirum.server --zoo <local zoo path> [--port <server TCP port>]
```

Keyword Args:
    --zoo (str): Path to a local model zoo directory, containing AI models you want your AI server to serve.

        - One possible way to fill local model zoo directory is to download models from a model zoo repo using
        `download_models()` function and provide the path to the local zoo directory as `--zoo` parameter.

    --port (int): TCP port to bind server to.

        - Default is 8778.

- When AI server is started, it runs indefinitely.
- If you started the server from a terminal, you may press `Enter` to shut it down.
- If you started the server headless (for example, as a service), then to shut down the server you need to kill
the Python process which runs the server.

The module also exposes `download_models()` function which can be used to prepare local model zoo directory to be
served by AI server:

- You first download models from the model zoo repo of your choice into some local directory of your choice by
calling [degirum.server.download_models][] function.
- Then you start the AI server providing the path to that local directory as `--zoo` parameter.

"""

import argparse
from pathlib import Path

from .zoo_manager import ZooManager
from ._zoo_accessor import _CloudServerZooAccessor

_default_port = 8778


def download_models(
    path: str, *, url: str = ZooManager._default_cloud_url, token: str = "", **kwargs
):
    """Download all models from a model zoo repo specified by the URL.

    Args:
        path: Local filesystem path to store models downloaded from a model zoo repo.

        url: Zoo repo URL.

            - If not specified, then DeGirum public model zoo URL will be used.

        token: Zoo repo authorization token.

    Keyword Args:
        model_family (str): Model family name filter.

            - When you pass a string, it will be used as search substring in the model name. For example, `"yolo"`, `"mobilenet"`.
            - You may also pass `re.Pattern`. In this case it will do regular expression pattern search.

        device (str): Target inference device -- string or list of strings of device names.

            - If passed, only models targeting the specified device(s) will be downloaded.

        precision (str): Model calculation precision -- string or list of strings of model precision labels.

            - Possible labels: `"quant"`, `"float"`.

        pruned (str): Model density -- string or list of strings of model density labels.

            - Possible labels: `"dense"`, `"pruned"`.

        runtime (str): Runtime agent type -- string or list of strings of runtime agent types.

            - Possible types: `"n2x"`, `"tflite"`, `"tensorrt"`, `"openvino"`.
    """
    root_path = Path(path)
    root_path.mkdir(exist_ok=True)
    zoo = _CloudServerZooAccessor(url=url, token=token)
    print(f"Downloading models\n  from '{url}'\n  into '{path}'")
    n = 0
    for m in zoo.list_models(**kwargs):
        zoo.download_model(m, root_path)
        print(m)
        n += 1
    print(f"Downloaded {n} model(s)")


def _download_zoo_run(args):
    """
    Download all models from a model zoo repo

    Args:
        args: argparse command line arguments
    """

    download_models(
        args.path,
        url=args.url,
        token=args.token,
        model_family=args.model_family,
        device=args.device,
        runtime=args.runtime,
        precision=args.precision,
        pruned=args.pruned,
    )


def _download_zoo_args(parser):
    """
    Define download-zoo subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "--path",
        default=".",
        help="local filesystem path to store models downloaded from a model zoo repo",
    )
    parser.add_argument(
        "--url",
        default=ZooManager._default_cloud_url,
        help="cloud model zoo URL; if not specified then DeGirum public model zoo URL will be used",
    )
    parser.add_argument(
        "--token",
        help="model zoo repo authorization token",
    )
    parser.add_argument(
        "--model_family",
        help="model family name filter: model name substring or regular expression",
    )
    parser.add_argument(
        "--device",
        help="target inference device filter",
    )
    parser.add_argument(
        "--runtime",
        help="runtime agent type filter",
        choices=["N2X", "TFLITE", "ONNX", "OPENVINO", "TENSORRT", "RKNN"],
    )
    parser.add_argument(
        "--precision",
        help="model calculation precision filter",
        choices=["QUANT", "FLOAT"],
    )
    parser.add_argument(
        "--pruned",
        help="model density filter",
        choices=["PRUNED", "DENSE"],
    )
    parser.set_defaults(func=_download_zoo_run)


def _server_run(args):
    """
    Start and run AI server

    Args:
        args: argparse command line arguments
    """
    from dataclasses import dataclass

    @dataclass
    class ServerParams:
        protocol: str
        port: int

    servers = []
    port = args.port if args.port > 0 else _default_port
    if args.protocol == "asio" or args.protocol == "both":
        servers.append(ServerParams("asio", port))
        port += 1
    if args.protocol == "http" or args.protocol == "both":
        servers.append(ServerParams("http", port))

    def localhost(server: ServerParams):
        return (
            "http://" if server.protocol == "http" else ""
        ) + f"127.0.0.1:{server.port}"

    if args.command == "start":
        import threading
        from contextlib import ExitStack
        from .CoreClient import Server

        server_object_list = []

        with ExitStack() as stack:
            for server in servers:
                server_object = stack.enter_context(
                    Server(server.port, args.zoo, server.protocol)
                )
                server_object.start()
                server_object_list.append(server_object)

            def keyboard_handler():
                try:
                    msg = ""
                    for server in servers:
                        msg += f"DeGirum {server.protocol} server is started at TCP port {server.port}\n"

                    input(
                        msg
                        + f"Local model zoo is served from '{args.zoo}' directory.\n"
                        + "Press Enter to stop the server\n"
                    )

                    for server_object in server_object_list:
                        server_object.stop(False)  # no wait
                    print("Requesting server to stop...")
                except BaseException:
                    pass

            if not args.quiet:
                threading.Thread(
                    target=keyboard_handler,
                    name="dg-server-keyboard_handler",
                    daemon=True,
                ).start()

            try:
                for server_object in server_object_list:
                    server_object.wait()
            except BaseException:
                pass

            if not args.quiet:
                print("\nServer stopped")

    elif args.command == "rescan-zoo":
        # send rescan model zoo command to server on localhost
        from .aiclient import zoo_manage

        for server in servers:
            zoo_manage(localhost(server), {"rescan": 1})

    elif args.command == "shutdown":
        # send shutdown command to server on localhost
        from .aiclient import shutdown

        for server in servers:
            shutdown(localhost(server))

    elif args.command == "cache-dump":
        # send cache dump command to server on localhost
        from .aiclient import zoo_manage

        for server in servers:
            print("Dumping cache for server: ", server)
            print(zoo_manage(localhost(server), {"cache_dump": 1}))


def _server_args(parser):
    """
    Define server subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """
    parser.add_argument(
        "command",
        nargs="?",
        choices=["start", "rescan-zoo", "shutdown", "cache-dump"],
        default="start",
        help="server command: start server; rescan local server model zoo; shutdown local server, dump agent cache info",
    )
    parser.add_argument(
        "--port", type=int, default=_default_port, help="[start] TCP port to bind to"
    )
    parser.add_argument(
        "--protocol",
        choices=["asio", "http", "both"],
        default="asio",
        help="[start] server protocol to use; use `both` to have `asio` and `http` protocols on two consecutive ports",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="[start] do not display any output"
    )
    parser.add_argument(
        "--zoo", default=".", help="[start] model zoo directory to serve models from"
    )
    parser.set_defaults(func=_server_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"{__file__}", description="DeGirum AI Server starter"
    )
    _server_args(parser)
    _server_run(parser.parse_args())
