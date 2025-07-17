#
# _runtime_installer.py - DeGirum Python SDK: runtime and driver installer
# Copyright DeGirum Corp. 2025
#
# Contains DeGirum command line runtime installation
#

import argparse
import os
import platform
import subprocess


def _get_platform():
    """
    Get OS/platform of current system
    """

    current_platform = platform.system()
    if current_platform == "Linux":
        is_debian = False
        try:
            with open("/etc/os-release") as f:
                os_release = f.read().lower()
                is_debian = "debian" in os_release or "ubuntu" in os_release or "raspbian" in os_release
        except FileNotFoundError:
            is_debian = False
        return "Debian" if is_debian else "Other Linux"
    elif current_platform == "Windows":
        return current_platform
    elif current_platform == "Darwin":
        return "macOS"
    else:
        return "Unknown OS"


def _run_script(script_path, args_list, sys_platform):
    """
    Execute shell or batch script with additional args

    Args:
        script_path: path to executable
        args_list: list of additional command line args (may be empty)
        sys_platform: OS of system. Currently supported: (Debian, Windows)
    """

    try:
        if sys_platform == "Debian":
            subprocess.run(["bash", script_path] + args_list, check=True)
        elif sys_platform == "Windows":
            subprocess.run(["cmd.exe", "/c", script_path] + args_list, check=True)
        else:
            print(f"Platform \"{sys_platform}\" not supported for script \"{script_path}\"")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        return


def _runtime_installer(args):
    """
    Execute install-runtime command

    Args:
        args: argparse command line arguments
    """

    current_platform = _get_platform()  # Platform of current system
    current_file_path = os.path.abspath(__file__)  # Full path of this file
    current_dir = os.path.dirname(current_file_path)  # Directory containing this file
    runtime_dir = os.path.join(current_dir, "runtime_installers")  # Runtime installer directory

    installer_suffix = ""  # OS-based installer file
    if current_platform == "Debian":
        installer_suffix = "sh"
    elif current_platform == "Windows":
        installer_suffix = "bat"

    # If command = "degirum install-runtime list", list plugins available for install
    if args.plugin_name == "list":
        print("Runtimes and versions available to install:")

        plugins_list = []

        for entry in os.scandir(runtime_dir):
            if entry.is_dir():
                install_script = os.path.join(entry.path, f"install_{entry.name}.{installer_suffix}")
                if os.path.isfile(install_script):
                    plugins_list.append(install_script)

        # Print plugins dics
        for plugin in plugins_list:
            _run_script(plugin, ["--list"], current_platform)

    # Else check for runtime_dir valitidy and run installation script
    else:
        install_script_path = os.path.join(runtime_dir, args.plugin_name, f"install_{args.plugin_name}.{installer_suffix}")

        # Check existence of install script
        if (not os.path.isfile(install_script_path)):
            print(f"\"{args.plugin_name}\" is not a valid plugin name. "
                  f"Try \"degirum install-runtime list\" to see a list of plugins available for install.")
            return

        # Parse extra args, set to empty list if none
        plugin_args_list = args.plugin_args if args.plugin_args else []

        # Run the install script
        _run_script(install_script_path, plugin_args_list, current_platform)


def _install_runtime_args(parser):
    """
    Define install subcommand arguments

    Args:
        parser: argparse parser object to be stuffed with args
    """

    parser.add_argument(
        "plugin_name",
        help="Plugin name to install, or 'list' to show available plugins"
    )
    parser.add_argument(
        "plugin_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the plugin-specific install script. Run \"degirum install-runtime <plugin name> --help to see plugin-specific options.\"")

    parser.set_defaults(func=_runtime_installer)
