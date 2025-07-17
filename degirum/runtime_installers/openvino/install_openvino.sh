#!/bin/bash

# This script installs OpenVINO on Ubuntu 20.04.
# Script will not overwrite existing installation of the same version.

SUPPORTED_VERSIONS=("2023.3.0" "2024.2.0" "2024.6.0")

declare -A openvino_versions
openvino_versions["2024.6.0"]="https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu20_2024.6.0.17404.4c0f47d2335_x86_64.tgz"
openvino_versions["2024.2.0"]="https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.2/linux/l_openvino_toolkit_ubuntu20_2024.2.0.15519.5c0f38f83f6_x86_64.tgz"
openvino_versions["2023.3.0"]="https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz"

######################################################################
# Enclosed area should be as uniform as possible between plugins

# Color definition
R='\033[0;31m' # Red
G='\033[0;32m' # Green
B='\033[4;34m' # Underlined blue
M='\033[1;35m' # Bright magenta
N='\033[0m'    # No color (reset)

# Help statement parsing
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    cat <<HELPDOC
usage: degirum install-runtime openvino [ version | --help | --list ]

Installs OpenVINO runtime libraries and device driver

positional arguments:
  version         Version of plugin to install, -a or -all to install all available

options:
  -h, --help      show this help message and exit
  -l, --list      list available versions for installation
HELPDOC
    exit 0  # Exit after printing
fi

# List statement parsing
    if [[ "$1" == "-l" || "$1" == "--list" ]]; then
    for VERSION in "${SUPPORTED_VERSIONS[@]}"; do
        echo " openvino $VERSION"
    done
    exit 0  # Exit after printing
fi

# Search for version in SUPPORTED_VERSIONS
valid_version=false
selected_versions=("none")
for VERSION in "${SUPPORTED_VERSIONS[@]}"; do
    if [[ "$VERSION" == "$1" ]]; then
        valid_version=true
        selected_versions[0]=$1
        break
    fi
done

# Handle --all
if [[ "$1" == "-a" || "$1" == "--all" ]]; then
    valid_version=true
    selected_versions=("${SUPPORTED_VERSIONS[@]}")
fi

# Check if valid version is selected
if ! $valid_version; then
    echo -e "${R}\"$1\" is not a valid version for plugin ${B}openvino${N}"
    exit 1
fi

# Confirm to user what is being installed
echo -e "${G}Installing plugin: ${B}openvino${G}, version: ${B}${selected_versions[*]}${N}"

# Prompt the user for sudo access
sudo --validate

######################################################################

install_openvino() {
    version=$1
    link=$2
    dir="/opt/intel/openvino_$version"

    # Check if the version is already installed
    if [ -d "$dir" ]; then
        echo -e "${B}OpenVINO${G} version ${B}$version${G} is already installed.${N}"
        return
    fi

    # Create /opt/intel if it doesn't exist
    sudo mkdir -p /opt/intel

    # Download and extract OpenVINO
    echo -e "${M}Downloading ${N}${B}OpenVINO${M}...${N}"
    curl -L "$link" --output "openvino_$version.tgz"
    tar -xf "openvino_$version.tgz"
    extracted_folder=$(tar -tf "openvino_$version.tgz" | head -1 | cut -f1 -d"/")
    sudo mv "$extracted_folder" "$dir"

    # Install dependencies
    echo -e "${M}Installing dependencies for ${N}${B}OpenVINO${M}...${N}"
    sudo -E "$dir/install_dependencies/install_openvino_dependencies.sh"

    # Remove the downloaded tar file
    rm "openvino_$version.tgz"

    # Success message
    echo -e "${B}OpenVINO${G} version ${B}$version${G} installed successfully.${N}"
}

for version in "${selected_versions[@]}"; do
    install_openvino "$version" "${openvino_versions[$version]}"
done
