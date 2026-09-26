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

A Pi 4B without a heatsink or fan reaches about 73 °C with the station
running and nobody lifting. It slows itself down at 80 °C. Fit the fan for a
training session.

## Install

Start with a current 64-bit Raspberry Pi OS installation and make sure the
camera is visible before installing this repository:

```sh
sudo apt update
sudo apt install -y git python3-venv python3-picamera2 ffmpeg

rpicam-hello --list-cameras
```

Create a virtual environment. `--system-site-packages` is intentional:
Raspberry Pi OS supplies `picamera2`, `libcamera` and `numpy` through apt, and
an isolated virtualenv would not be able to import them.

```sh
python3 -m venv --system-site-packages ~/kvihtai-venv
~/kvihtai-venv/bin/pip install "fastapi==0.141.1" "uvicorn[standard]==0.53.0" \
    "pydantic-settings>=2.0.0" "opencv-python-headless==5.0.0.93"
```

This installs only what the station needs, and keeps the system `numpy` that
`picamera2` was built against, rather than the pinned one in
`requirements.txt`.

## Copy the code over

The Pi does not need a git checkout. From a machine with this repository and
[kvihtai-web](https://github.com/mimikunda/kvihtai-web) side by side, build the
web app and copy both over:

```sh
(cd ../kvihtai-web && npm ci && npm run build)
deploy/push.sh david@192.168.1.174
```

This copies `app/`, `tools/`, `deploy/` and the web app's `dist/` to
`~/kvihtai-core` and `~/kvihtai-web` on the Pi, and restarts the station if it
is installed. Data on the Pi is not touched.

## Install as a boot service

On the Pi, once:

```sh
VENV=~/kvihtai-venv bash ~/kvihtai-core/deploy/setup_pi.sh
```

It needs `sudo`, and sets up:

- **`kvihtai.service`**, the station: camera, set detection, recording, the
  API and the web app, on port 8080. It starts at boot and restarts itself.
- **`kvihtai-network.service`** and a NetworkManager connection
  `kvihtai-hotspot`. At boot the Pi waits 40 s for a Wi-Fi it knows. If none
  comes, it opens the Wi-Fi network `KvihtAI`, and the station is at
  `http://10.42.0.1:8080`, or `http://kvihtai.lan:8080`.

The script prints the hotspot's password. It keeps the existing one when run
again; set `KVIHTAI_HOTSPOT_PASSWORD` to choose one.

To switch by hand:

```sh
bash ~/kvihtai-core/deploy/network.sh hotspot    # open the hotspot now
bash ~/kvihtai-core/deploy/network.sh wifi       # back to a known Wi-Fi
```

Logs:

```sh
journalctl -u kvihtai -f
```

For the station to set the clock from a phone when there is no internet, the
service's user needs `sudo` without a password for `date`. Without it, the
station keeps the difference to the phone's clock itself, and only the names
of the recordings made before the phone connected are off.

## At the gym

1. Stand the camera side-on to the bar and power the Pi. Give it a minute.
2. On the phone, join the Wi-Fi `KvihtAI`. Android may warn that it has no
   internet; choose to stay connected.
3. Open `http://10.42.0.1:8080`. Adding it to the home screen gives it an icon.
4. On **Camera**, turn the picture upright, press **Focus**, and check that the
   plate on the bar is ringed in green while it waits.
5. Lift. The set shows on **Now** once it has been analysed.

## Where things are on the Pi

Under `~/kvihtai-data`:

- `recordings/`: everything the camera saw, in 5 minute segments of MP4, each
  with a JSON file beside it. About 5 GB an hour. The oldest are deleted only
  when less than 5 GB is free.
- `sets/<start time>/`: each set's `result.json` and `clip.mp4`.
- `kvihtai.db`: the sets the app shows.
- `camera.json`: rotation, exposure, gain and focus, as last set in the app.

Copy the recordings off with, for example:

```sh
rsync -av david@192.168.1.174:kvihtai-data/recordings/ footage/pi/
```

## Run by hand

The station can run in the foreground, with the service stopped:

```sh
sudo systemctl stop kvihtai
cd ~/kvihtai-core
KVIHTAI_CAMERA=pi KVIHTAI_DATA_DIR=~/kvihtai-data ~/kvihtai-venv/bin/python -m uvicorn app.main:app --port 8080
```

`KVIHTAI_CAMERA=video:path/to/clip.mp4` plays a recording over and over in
place of the camera, on the Pi or on a laptop. The command-line capture tool
still works on its own, without the API:

```sh
~/kvihtai-venv/bin/python tools/capture.py --pi --size 1536x864 --fps 80 \
  --exposure 2000 --gain 8 --out sets/
~/kvihtai-venv/bin/python tools/bench_capture.py path/to/clip.mp4
```

## What works now

- The station starts at boot, records everything the camera sees, detects
  sets, analyses them and serves them to the web app, with a clip of each set.
- On a Pi 4B it keeps 60 fps at 1536x864 while recording, losing 1 to 4 % of
  frames while the app's camera preview is open.
- The hotspot comes up when no known Wi-Fi is in range. It was checked from the
  Pi's side: the network is visible and hands out addresses. A phone has not
  joined it yet.
- The tracker and analysis were verified on two handheld phone clips and
  synthetic plates. Nothing has been lifted in front of the Pi camera yet.
- The live stream at `/ws/live` follows `docs/API_CONTRACT.md` while a set goes
  on. `/api/v1/sets/latest` returns the newest set with one rep per rise.

## What does not work or is not finished

- There is no packaged installer, Pi image, or Pi-specific dependency lockfile.
- On a Pi 4B, the analysis takes about half a minute for a short set, and a
  long set with the plate near the camera may lose frames live; see the open
  questions in [DESIGN.md](DESIGN.md).
- Views much beyond about 25 degrees off square are not supported reliably;
  the design notes call out failures around 30 to 50 degrees.
- Absolute scale has not yet been checked against an independent measurement.
- Handheld camera movement contaminates the bar path. Use a tripod for useful
  measurements.

For implementation details and benchmark data, see [docs/DESIGN.md](DESIGN.md).
