# AI4DrugDesign Spring 2026

Interactive drug discovery pipeline built with Gradio. Enter a PDB protein ID and the app runs: protein analysis with 3D visualization, compound discovery from ChEMBL, Lipinski Rule of 5 filtering, molecular docking with AutoDock Vina, and ADME filtering.

## Setup

### macOS

1. Install [Homebrew](https://brew.sh) if you don't have it, then:
   ```bash
   brew install swig boost
   ```

2. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Restart your terminal after installing.

3. Install dependencies:
   ```bash
   export CPLUS_INCLUDE_PATH="$(brew --prefix boost)/include"
   export LIBRARY_PATH="$(brew --prefix boost)/lib"
   uv sync
   ```

### Windows

1. Open PowerShell as administrator and run:
   ```powershell
   wsl --install
   ```
   Restart your computer when prompted.

2. Open the **Ubuntu** app from the Start menu. It will ask you to create a username and password (this is just for the Linux environment).

3. Inside the Ubuntu terminal, install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

4. Install dependencies:
   ```bash
   uv sync
   ```
   No extra system libraries needed — Linux has pre-built AutoDock Vina packages.

## Running

```bash
uv run python app.py
```

Opens at http://localhost:7860.
