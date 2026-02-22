# HAIDEF MedGamma LoRA GPU Setup Instructions

## 1. Recommended Environment
- OS: Windows 10/11 (64-bit)
- Python: 3.10 or 3.11 (do NOT use 3.13)
- NVIDIA GPU: Quadro T1000 or similar
- CUDA Toolkit: 11.8 (matching PyTorch CUDA version)

## 2. Create Conda Environment
```
conda create -n haidef-gpu python=3.11
conda activate haidef-gpu
```

## 3. Install CUDA Toolkit (if not already installed)
- Download and install CUDA 11.8 from NVIDIA: https://developer.nvidia.com/cuda-toolkit-archive
- Add CUDA to your PATH if not auto-detected.

## 4. Install PyTorch with CUDA
```
pip install torch==2.6.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## 5. Install All Required Libraries
```
pip install -r requirements.txt
```

## 6. Verify GPU Availability
```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## 7. Run LoRA Training
```
python scripts/train_lora.py
```

## 8. Troubleshooting
- If CUDA is not detected, check your CUDA installation and PATH.
- If package conflicts occur, ensure you are using Python 3.10 or 3.11 and the provided requirements.txt.
- For latest package versions, always use pip (not conda) for PyTorch and related libraries.

## 9. Useful Links
- PyTorch CUDA: https://pytorch.org/get-started/locally/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive

---
For further help, contact your project maintainer or check official docs.
