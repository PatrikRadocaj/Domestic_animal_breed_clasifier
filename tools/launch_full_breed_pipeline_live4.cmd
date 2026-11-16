@echo off
setlocal
cd /d "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje"
start "" powershell.exe -NoExit -ExecutionPolicy Bypass -File "C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\tools\retrain_full_breed_pipeline_visible_amd.ps1" -Stage1Epochs 50 -Stage2Epochs 10 -DirectYoloEpochs 50 -RunStamp "live4_20260622_0406"
