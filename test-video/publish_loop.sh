#!/bin/sh
set -e

: "${MTX_RTSP_URL:?must set MTX_RTSP_URL, e.g. rtsp://mediamtx:8554/cam1}"
: "${VIDEO_FILE:?must set VIDEO_FILE, e.g. /media/video2.mp4}"

echo "[publisher] waiting for mediamtx to accept connections..."
sleep 3

echo "[publisher] publishing ${VIDEO_FILE} -> ${MTX_RTSP_URL} (looping forever)"

while true; do
    ffmpeg -re -stream_loop -1 -i "${VIDEO_FILE}" \
        -c:v copy -bsf:v h264_mp4toannexb -an \
        -f rtsp -rtsp_transport tcp \
        "${MTX_RTSP_URL}"
    echo "[publisher] ffmpeg exited (mediamtx restarting? file issue?) - retrying in 2s"
    sleep 2
done
