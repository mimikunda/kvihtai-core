# Raspberry Pi setup

This is the current, practical setup for running KvihtAI Core on a Raspberry
Pi. It describes the code as it exists today, not a promised production image.

## Hardware

The intended setup is:

- Raspberry Pi 5, or a Raspberry Pi 4B for development and benchmarking
- Raspberry Pi Camera Module 3
- Raspberry Pi OS with the `libcamera` camera stack
- A tripod-mounted camera looking at the bar from the side

The capture code uses `picamera2` and configures the camera directly. A USB
camera is not currently wired into the Pi capture path; recorded video is the
portable fallback and is useful for development on any machine.

## Install

Start with a current 64-bit Raspberry Pi OS installation and make sure the
camera is visible before installing this repository:

```sh
sudo apt update
sudo apt install -y git python3-venv python3-picamera2

libcamera-hello
```

Clone the repository and create a virtual environment. `--system-site-packages`
is intentional: Raspberry Pi OS supplies `picamera2` through apt, so an
isolated virtualenv would not be able to import it.

```sh
cd /home/pi
git clone <repository-url> kvihtai-core
cd kvihtai-core
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

If the pinned OpenCV or NumPy wheels are not available for the Pi's Python
version, use a compatible Raspberry Pi OS package or update the pin locally;
the repository does not currently provide a Pi-specific lockfile or image.

## Check the service

Run the API in the foreground first:

```sh
KVIHTAI_RELOAD=0 ./run.sh
```

From another terminal on the Pi, check it with:

```sh
curl http://127.0.0.1:8080/health
```

The default bind address is `0.0.0.0:8080`, so a client on the local network
can use `http://<pi-ip>:8080`. Set `KVIHTAI_HOST` and `KVIHTAI_PORT` when a
different bind address or port is needed.

## Run the camera pipeline

The camera entry point is currently a command-line capture tool, separate from
the FastAPI process:

```sh
.venv/bin/python tools/capture.py --pi --size 1536x864 --fps 80 \
  --exposure 2000 --gain 8 --out sets/
```

Useful options are `--seconds` for a short trial, `--keep-frames` for saved
debug crops, and `--ring` for the number of seconds kept for pre-roll. The
defaults are 80 fps, a 2 ms exposure, and analogue gain 8. Adjust exposure and
gain for the lighting before changing the tracker.

For a repeatable smoke test without a camera, use recorded footage instead:

```sh
.venv/bin/python tools/capture.py --video path/to/clip.mp4 --out sets/
.venv/bin/python tools/bench_capture.py path/to/clip.mp4
```

Each completed set is written below the output directory with a
`result.json`. Keep the camera still during a set; camera motion is currently
included in the measured bar path.

## Install as a boot service

The repository includes a systemd unit written for a user named `pi` and a
checkout at `/home/pi/kvihtai-core`:

```sh
sudo cp deploy/kvihtai.service /etc/systemd/system/kvihtai.service
sudo systemctl daemon-reload
sudo systemctl enable --now kvihtai.service
sudo systemctl status kvihtai.service
```

View logs with:

```sh
journalctl -u kvihtai.service -f
```

If the checkout or Linux user differs, edit the unit's `User`, `Group`,
`WorkingDirectory`, and virtualenv paths before enabling it. The unit runs one
Gunicorn worker and binds the API to port 8080.

## What works now

- The FastAPI application starts and exposes `/health`.
- The latest-set REST endpoint exists at `/api/v1/sets/latest` and returns 404
  when no set has been stored.
- Recorded video can stand in for the camera, which makes pipeline work
  possible without Pi hardware.
- The live capture pipeline supports Pi camera input through `picamera2`,
  including pre-roll, watched plates, set recording, and post-set analysis.
- The tracker and analysis were verified on two handheld phone clips and
  synthetic plates. The end-to-end capture pipeline was measured on a Pi 4B;
  both test clips kept up with their source rates. The documented Pi 5 speedup
  is an expectation, not a completed benchmark.
- The current tested geometry is a mostly square side view. The design notes
  report reliable synthetic tracking through about 25 degrees off square.

## What does not work or is not finished

- There is no packaged installer, Pi image, or Pi-specific dependency lockfile.
- The API process does not start the camera capture session. Run
  `tools/capture.py --pi` separately; the systemd unit currently manages only
  the FastAPI daemon.
- The WebSocket live endpoint described in `docs/API_CONTRACT.md` is not
  registered in the current router.
- Camera capture requires `picamera2` from the Pi OS environment and cannot be
  tested on a normal laptop without recorded footage.
- Views much beyond about 25 degrees off square are not supported reliably;
  the design notes call out failures around 30 to 50 degrees.
- Absolute scale has not yet been checked against an independent measurement.
- Handheld camera movement contaminates the bar path. Use a tripod for useful
  measurements.
- Long, high-frame-rate sets may take minutes to analyse after capture. The
  current Pi 4B measurements were 66 to 134 ms per frame for post-set analysis,
  depending on the clip.

For implementation details and benchmark data, see [docs/DESIGN.md](DESIGN.md).