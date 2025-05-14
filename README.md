# Autonomous Line-Following & Pick-and-Place Robot

**raspbbery pi, fully-autonomous robot** that follows a line, localizes itself on a node-based grid, and picks & places boxes within 90 s per cycle. We’re just getting started, and this repo will house every design decision, line of code, and simulation model as we build toward our first working prototype.

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

1. **Follow** a dark line on with a web cam.
3. **Detect** and **grasp** boxes via the camera and servo-driven gripper.
4. **Transport** boxes between pint a to point b in under **90 s**.

All control logic runs on an **Pi5** under **MicroPython(or just pyhon)**, and we validate core behaviors in **Webots** before moving to hardware. We haven’t finished yet—we’re just getting started.

---

## 🚀 Features

* **Robust Line-Following:** PID-based with broken-line recovery.
* **Box Identification:** Ensures we pick only the correct box.
* **Modular Firmware:** HAL layer abstracts hardware; FSM manages states.
* **Compact Design:** Constrained to a 30 × 30 × 30 cm envelope.
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
* Git 2.25+
* KiCad
* Fusion 360 / FreeCAD/ Solidworks
* Webots R2023b

### Clone & Install

```bash
git clone git@github.com:Simonf8/autonomous-linefollower-robot.git
cd autonomous-linefollower-robot
git config --global credential.helper cache   # optional
```

### Flash Firmware

Connect the to your raspbbery pi with ssh.

