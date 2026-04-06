Set-Location "C:\dev\trading"

& ".\.venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall `
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 `
  --index-url https://download.pytorch.org/whl/cu128
