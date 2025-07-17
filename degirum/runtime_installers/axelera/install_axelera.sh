#!/bin/bash

# This script installs Axelera runtime and Metis driver on any Debian based system.
# Script will overwrite any existing installation of Metis drivers

set -e

SUPPORTED_VERSIONS=("1.3.1")

declare -A METIS_VERSIONS
METIS_VERSIONS["1.3.1"]="1.0.2"

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
usage: degirum install-runtime axelera [ version | --help | --list ]

Installs Axelera runtime libraries and device driver

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
        echo " axelera $VERSION"
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
    echo -e "${R}\"$1\" is not a valid version for plugin ${B}axelera${N}"
    exit 1
fi

# Confirm to user what is being installed
echo -e "${G}Installing plugin: ${B}axelera${G}, version: ${B}${selected_versions[*]}${N}"

# Prompt the user for sudo access
sudo --validate

######################################################################

install_axelera() {
    AXELERA_VERSION=$1
    METIS_VERSION=$2

    if [ -d "/opt/axelera/runtime-$AXELERA_VERSION-1" ]; then
        echo -e "${B}Axelera${G} version ${B}$AXELERA_VERSION${G} is already installed.${N}"
        return
    fi

    # Set variables
    GPG_URL="https://software.axelera.ai/artifactory/api/security/keypair/axelera/public"
    GPG_KEY_PATH="/etc/apt/keyrings/axelera.gpg"
    REPO_LIST_PATH="/etc/apt/sources.list.d/axelera.list"
    REPO_SOURCE="https://software.axelera.ai/artifactory/axelera-apt-source/ stable main"

    # Create keyring directory if it doesn't exist
    sudo mkdir -p "$(dirname "$GPG_KEY_PATH")"

    # Download and store the GPG key
    curl -fsSL "$GPG_URL" | gpg --dearmor | sudo tee "$GPG_KEY_PATH" > /dev/null

    # Ensure correct permissions for the GPG key
    sudo chmod 644 "$GPG_KEY_PATH"

    # Add the repository source
    echo "deb [signed-by=$GPG_KEY_PATH] $REPO_SOURCE" | sudo tee "$REPO_LIST_PATH" > /dev/null

    # Update APT, ignore errors encountered by apt
    sudo apt update || true

    echo -e "${M}Downloading packages, this may take a while...${N}"

    # Install the packages with specific versions
    sudo apt install -y \
        axelera-runtime-$AXELERA_VERSION \
        axelera-device-$AXELERA_VERSION \
        metis-dkms=$METIS_VERSION \
        axelera-riscv-gnu-newlib-toolchain-409b951ba662-7

    # Add Axelera runtime libraries to ldconfig
    echo "/opt/axelera/runtime-$AXELERA_VERSION-1/lib" | sudo tee /etc/ld.so.conf.d/axelera.conf > /dev/null
    sudo ldconfig

    echo -e "${B}Axelera${G} version ${B}$AXELERA_VERSION${G} installed successfully.${N}"
} 

for version in "${selected_versions[@]}"; do
    install_axelera "$version" "${METIS_VERSIONS[$version]}"
done
