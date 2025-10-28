# Black Hole Gravitational Lensing Simulation

This project contains three distinct simulations that visualize how black holes bend spacetime and affect the paths of light rays (photons):

1. **2D Simulation** (`blackhole2D.py`) - Interactive 2D visualization of light ray trajectories
2. **3D Real-time Simulation** (`blackhole3D.py`) - Interactive 3D gravitational lensing with accretion disk
3. **3D Video Recorder** (`blackhole3D_video.py`) - High-quality video generation of cinematic black hole sequences

Video: [https://youtu.be/ApUE-OKH7cM](https://youtu.be/ApUE-OKH7cM)


## Physics Accuracy

For a detailed explanation of the physics and equations used in these simulations, see [physics.md](physics.md).

- **Schwarzschild Metric**: Accurate geodesic calculations in curved spacetime
- **Runge-Kutta Integration**: 4th-order numerical integration for precise ray tracing
- **Gravitational Lensing**: Realistic light bending effects around massive objects
- **Event Horizon**: Proper black hole boundary where light cannot escape
- **Photon Sphere**: Critical orbit radius at 1.5× Schwarzschild radius


## Acknowledgments

- Huge thanks to kavan for his awesome [video](https://www.youtube.com/watch?v=8-B6ryuBkCM) and [repo](https://github.com/kavan010/black_hole) on black hole simulation in C++. I tried making my own version before but couldn't get it right. His project showed me the way.
- Also, shoutout to Cursor and Claude-4-sonnet for writing most of this code.


## Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/fajrulfx/blackhole
cd blackhole
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run a simulation:**

```bash
# 2D Interactive Simulation
python blackhole2D.py

# 3D Real-time Simulation with GPU
python blackhole3D.py

# command line options
python blackhole3D.py --width 1920 --height 1080    # Set resolution
python blackhole3D.py --resolution 1920x1080        # Alternative format
python blackhole3D.py --grid                        # Enable grid overlay

# 3D Video Recording
python blackhole3D_video.py

# command line options
python blackhole3D_video.py --width 1920 --height 1080 --fps 30 --duration 30
python blackhole3D_video.py --resolution 4K --fps 60 --duration 60

```
