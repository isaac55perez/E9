# Setup Instructions

This document provides step-by-step instructions for setting up your environment to run the convolution exercise.

## Python Environment Setup

1. **Python Installation**
   - Install Python 3.10 or higher from [python.org](https://python.org)
   - Ensure Python is added to your system PATH during installation

2. **Virtual Environment Creation**
   ```bash
   # Navigate to the project directory
   cd path/to/project

   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # For Windows CMD:
   .venv\Scripts\activate
   # For Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # For Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   # Make sure your virtual environment is activated
   pip install numpy matplotlib scipy
   ```

## Terminal Configuration

### Windows CMD

1. Open Command Prompt
2. Navigate to project directory:
   ```cmd
   cd path\to\project
   ```
3. Activate virtual environment:
   ```cmd
   .venv\Scripts\activate
   ```

### Windows PowerShell

1. If you haven't already, enable script execution:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSignedss -Scope CurrentUser
   ```
2. Navigate to project directory:
   ```powershell
   cd path\to\project
   ```
3. Activate virtual environment:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

### WSL (Windows Subsystem for Linux)

1. Open WSL terminal
2. Navigate to project directory:
   ```bash
   cd /mnt/drive-letter/path/to/project
   ```
3. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Troubleshooting

If you encounter any issues:

1. **ModuleNotFoundError**:
   - Ensure virtual environment is activated (you should see (.venv) in your prompt)
   - Reinstall packages:
     ```bash
     pip install --force-reinstall numpy matplotlib scipy
     ```

2. **Permission Issues**:
   - For PowerShell, check execution policy
   - For WSL, ensure proper file permissions

3. **Display Issues**:
   - If running in WSL, ensure X11 forwarding is configured
   - For remote sessions, use `plt.savefig()` instead of `plt.show()`