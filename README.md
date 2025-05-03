# Autonomous Line-Following & Pick-and-Place Robot

**ESP32-powered, fully-autonomous robot** that follows a line, localizes itself on a node-based grid, and picks & places boxes within 90 s per cycle. We’re just getting started, and this repo will house every design decision, line of code, and simulation model as we build toward our first working prototype.

---

## 📋 Table of Contents

* [📖 Overview](#-overview)
* [🚀 Features](#-features)
* [🌐 Simulation](#-simulation)
* [📂 Repository Structure](#-repository-structure)
* [⚙️ Getting Started](#-getting-started)

  * [Prerequisites](#prerequisites)
  * [Clone & Install](#clone--install)
  * [Flash Firmware](#flash-firmware)
* [🛠 Development Workflow](#-development-workflow)
* [🤖 Hardware Overview](#-hardware-overview)
* [💾 Firmware Overview](#-firmware-overview)
* [📣 Team & Roles](#-team--roles)
* [📝 Documentation & Compliance](#-documentation--compliance)
* [📄 License](#-license)

---

## 📖 Overview

We’ve created a centralized repository to design, simulate, build, program, and document an autonomous line-following robot with pick-and-place capability. Our goals are to:

1. **Follow** a dark line on a light field using IR reflectance sensors.
2. **Localize** at nodes marked by reflectance or color beacons.
3. **Detect** and **grasp** boxes via a color sensor and servo-driven gripper.
4. **Transport** boxes between nodes in under **90 s**.

All control logic runs on an **ESP32** under **MicroPython**, and we validate core behaviors in **Webots** before moving to hardware. We haven’t finished yet—we’re just getting started.

---

## 🚀 Features

* **Robust Line-Following:** PID-based with broken-line recovery.
* **Node-Based Localization:** Unique reflectance markers answer “Where am I?”
* **Color-Based Box Identification:** Ensures we pick only the correct box.
* **Modular Firmware:** HAL layer abstracts hardware; FSM manages states.
* **Compact Design:** Constrained to a 30 × 30 × 30 cm envelope.
* **CE-Style Documentation:** DHF, risk assessments, and test reports.
* **Webots Simulation:** Digital twin for rapid iteration.

---

## 🌐 Simulation

We use **Webots** to model our robot and test line-following, localization, and pick-and-place logic in a virtual environment before hardware validation.

1. Open the Webots project in `simulation/`.
2. Launch the world:

   ```bash
   webots simulation/linefollower_world.wbt
   ```

Webots builds and runs `simulation/simulation_controller.py`, mirroring `firmware/main.py`.

Tweak PID gains and sensor positions live via Webots’ parameter fields.

Record cycle times and positional error with the built-in LogView.

We’re still refining our simulation; stay tuned for updated worlds and controller scripts.

---

## 📂 Repository Structure

```plaintext
/
├── README.md
├── .gitignore
├── docs/                   
│   ├── DHF/                
│   ├── standards/          
│   ├── meeting-notes/      
│   └── schematics/         
├── firmware/               
│   ├── hal/                
│   ├── pid/                
│   ├── localization/       
│   └── main.py             
├── hardware/               
│   ├── pcb/                
│   ├── mech/               
│   └── BOM.xlsx            
├── tests/                  
│   ├── firmware-unit/      
│   ├── integration/        
│   └── sensor-calibration/
└── simulation/             
    ├── linefollower_world.wbt
    ├── robot_proto.wbo
    └── simulation_controller.py
```

---

## ⚙️ Getting Started

### Prerequisites

* Python 3.10+
* mpremote
* Git 2.25+
* KiCad
* Fusion 360 / FreeCAD
* Webots R2023b

### Clone & Install

```bash
git clone git@github.com:Simonf8/autonomous-linefollower-robot.git
cd autonomous-linefollower-robot
git config --global credential.helper cache   # optional
```

### Flash Firmware

Connect the ESP32 via USB.

Install mpremote:

```bash
pip install mpremote
```

Deploy the firmware:

```bash
mpremote fs cp firmware/ /flash/
mpremote run firmware/main.py
```
