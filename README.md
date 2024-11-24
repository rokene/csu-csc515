# Computer vision (csu-csc515)

Computer vision

# Prerequisites

* WSL
* VcXsrv - Open-Source X Server for Windows https://vcxsrv.com/
* Python 3.10

# Setup

GTK support: `sudo apt install -y libgtk2.0-dev pkg-config libcanberra-gtk-module libcanberra-gtk3-module`

> WSL may already provide default x display forwarding configurations

Add x display forwarding into `.bashrc`: `export DISPLAY=$(ip route | awk '/default/ {print $3}'):0 # forwarding display`

# Usage

For options `make`

## Basic CV2 Project

Setup: `basic-cv2-setup`

Execute: `basic-cv2`

## Bank Note Analysis

Setup: `make banknote-setup`

Execute: `make banknote`

## Puppy Color Analysis

Setup: `make puppy-colors-setup`

Execute: `make puppy-colors`
