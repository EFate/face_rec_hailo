#!/bin/bash

# This script installs ONNX runtime on the target linux system into /usr/local/
# Script will overwrite existing installation.

SUPPORTED_VERSIONS=("1.20.1")

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
usage: degirum install-runtime onnx [ version | --help | --list ]

Installs ONNX runtime library

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
        echo " onnx $VERSION"
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
    echo -e "${R}\"$1\" is not a valid version for plugin ${B}onnx${N}"
    exit 1
fi

# Confirm to user what is being installed
echo -e "${G}Installing plugin: ${B}ONNX${G}, version: ${B}${selected_versions[*]}${N}"

# Prompt the user for sudo access
sudo --validate

######################################################################

install_onnx() {
    # Install ONNX runtime
    ONNX_RUNTIME_VERSION=$1
    arch_suff=$2
    if [ -d "/usr/local/onnxruntime-linux-$arch_suff-${ONNX_RUNTIME_VERSION}" ]; then
        echo -e "${B}ONNX${G} version ${B}$ONNX_RUNTIME_VERSION${G} is already installed.${N}"
        return
    fi
    curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-$arch_suff-${ONNX_RUNTIME_VERSION}.tgz
    tar -xzf onnxruntime-linux-$arch_suff-${ONNX_RUNTIME_VERSION}.tgz
    sudo mv onnxruntime-linux-$arch_suff-${ONNX_RUNTIME_VERSION} /usr/local
    sudo rm onnxruntime-linux-$arch_suff-${ONNX_RUNTIME_VERSION}.tgz
    echo -e "${B}ONNX${G} installed successfully.${N}"
}

# Determine architecture
if [[ "$(uname -m)" == "x86_64" ]]; then
    arch_suff=x64
    echo "Detected architecture: x86_64"
else
    arch_suff=aarch64
    echo "Detected architecture: ARM64"
fi

for version in "${selected_versions[@]}"; do
    install_onnx "$version" "$arch_suff"
done
