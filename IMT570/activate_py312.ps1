# Set the directory
$directory = "S:\code\uw\IMT570\py312_venv"

# Check if the directory exists
if (Test-Path -Path $directory -PathType Container) {
    # Change to the directory
    cd $directory

    # Check if the virtual environment exists (adjust the activation script name if needed)
    if (Test-Path -Path ".\Scripts\activate.ps1" -PathType Leaf) { # For venv
        # Activate the virtual environment (using the appropriate activation script)
        & ".\Scripts\Activate.ps1"  # or .\venv\Scripts\activate if you're not using powershell to activate the virtual env.
    }
    elseif (Test-Path -Path ".\.venv\Scripts\Activate.ps1" -PathType Leaf) { # For .venv
        & ".\.venv\Scripts\Activate.ps1" # or .\.venv\Scripts\activate if you're not using powershell to activate the virtual env.
    }
        elseif (Test-Path -Path ".\env\Scripts\Activate.ps1" -PathType Leaf) { # For env
        & ".\env\Scripts\Activate.ps1" # or .\env\Scripts\activate if you're not using powershell to activate the virtual env.
    }
    else {
        Write-Warning "Virtual environment activation script not found.  Make sure you have created and named your virtual environment correctly.  Common names are venv, .venv, or env."
        exit 1 # Exit the script with an error code
    }


    cd ..
    # Run Jupyter Notebook
    jupyter notebook

} else {
    Write-Warning "Directory '$directory' not found."
    exit 1 # Exit the script with an error code
}