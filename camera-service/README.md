# camera-service — the suite's single camera manager

A single `camera_stream` process serves every enabled module (face,
plate, heatmap, fire, or any `<module>:cameras:config` it finds).
MediaMTX itself is the `mediamtx` service in the root `compose.yaml`,
configured by `config/mediamtx.yml`.

- **Reads** each module's cameras from `<module>:cameras:config`.
  - It re-syncs when the backend publishes `<module>:camera:config:updated`, and every `CONFIG_RESYNC_SEC` (30 s).
- **Registers** one MediaMTX path per physical camera: `cam_<sha1(address)>` (`src/relay.py`).
  - A path is shared by every module that uses the camera.
  - It is removed when the last module stops using it.
  - Paths MediaMTX forgets (for example after a relay restart) are re-registered.
- **Watches** health per path, using MediaMTX readiness plus a TCP probe
  on the camera's own port, with an `OFFLINE_HOLD_SECONDS` debounce.
- **Writes** `<module>:cameras:details`: `connected`, `error`,
  `relay_path`, `stream_url`, plus the camera's config.
- **Publishes** each online/offline change to every module that uses the
  camera, on that module's own channel: `face:cameras:events`, or
  `<module>:camera:events` for the others. Override a channel with
  `CAMERA_EVENTS_CHANNEL_<MODULE>`.
- **Holds no state of its own:** on restart it rebuilds everything from Redis.

Test: `python3 tests/test_camera_stream.py`. It needs `redis-server`, and
uses a fake MediaMTX API.
