@echo off
rem Registers test camera 1 with a HEXAGON ROI (normalized points, deliberately scrambled
rem to prove the order is fixed by the detector) and activates it.
rem Run from this folder AFTER "docker compose up -d --build" shows the services healthy.
setlocal
if "%SHARED_NETWORK%"=="" set SHARED_NETWORK=eyeplate_net
if "%IMAGE_TAG%"=="" set IMAGE_TAG=dev
set RT=docker run --rm --network %SHARED_NETWORK% -e REDIS_URL=redis://redis:6379/0 -v "%cd%":/w -w /w eyeplate/plate-detector:%IMAGE_TAG% python redis_tools.py
%RT% set-camera --id 1 --address publisher --title "Hex test" --roi 0.5526 0.0954 0.3990 0.7565 --roi-polygon 0.87 0.8519 0.63 0.0954 0.5526 0.4737 0.9516 0.4737 0.63 0.8519 0.87 0.0954
%RT% activate --id 1
echo.
echo Camera 1 active. Check:  docker compose logs plate_detector --tail 100 ^| findstr "ROI PERF"
echo Debug video:        dir debug_video\1
