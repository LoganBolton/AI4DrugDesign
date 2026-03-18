# AI4DrugDesign_Spring2026

A simple Gradio-based application for AI4DrugDesign course.

## Installation

1. **Install uv** (if not already installed):

   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   After installation, restart your terminal or run:
   - macOS/Linux: `source $HOME/.local/bin/env` or `source $HOME/.cargo/env`
   - Windows: Restart PowerShell

   Verify installation:
   ```bash
   uv --version
   ```


2. **Install system build dependencies** (required to compile AutoDock Vina):

   **macOS (Homebrew):**
   ```bash
   brew install swig boost
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt install swig libboost-all-dev
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   This will:
   - Create a virtual environment in `.venv/`
   - Install Python 3.12 (specified in `.python-version`)
   - Install all dependencies from `pyproject.toml` (including AutoDock Vina)

## Running the App

```bash
uv run python app.py
```

To activate the virtual environment manually:
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`

## macOS Fix (AutoDock Vina Build Issues)
1. **Install Miniconda** (if not installed):

   **Apple Silicon (M1/M2/M3):** 

   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   bash Miniconda3-latest-MacOSX-arm64.sh
   source ~/.zshrc
   ```

   **Intel Mac:**

   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   bash Miniconda3-latest-MacOSX-x86_64.sh
   source ~/.zshrc
   ``` 
2. **Create and activate environment**

   ```bash
   conda create -n ai4drug312 python=3.12 -y
   conda activate ai4drug312
   ```
3. **Install compiled dependencies (avoids build failures)**

   ```bash
   conda install -c conda-forge numpy swig boost-cpp pip vina -y
   ``` 
   **Verify:**
   ```bash
   python -c "import vina; print(vina.__version__)"
   ```
4. **Navigate to the project directory**

   ```bash
   cd /path/to/AI4DrugDesign_Spring2026
   ``` 
5. **Install project (skip dependency rebuild)**

   ```bash
   pip install -e . --no-deps
   ```   
--no-deps prevents pip from rebuilding vina, which is already installed via Conda.

6. **Install remaining Python dependencies**

   ```bash
   pip install openai gemmi meeko prody psutil requests scipy gradio python-dotenv rdkit
   ```

7. **Run the app**

   ```bash
   python app.py
   ``` 
Then open:
http://127.0.0.1:7860

**Running the App (after setup)**

   ```bash
   conda activate ai4drug312
   cd /path/to/AI4DrugDesign_Spring2026
   python app.py
   ``` 
**After pulling new changes**

   ```bash
   conda activate ai4drug312
   cd /path/to/AI4DrugDesign_Spring2026
   git pull
   pip install -e . --no-deps
   python app.py
   ``` 