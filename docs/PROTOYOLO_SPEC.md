## ProtoYolo (Phylogenetic Perceptual Learner)

Design specification

This document describes a bio-mimetic vision system that begins as a high-speed,
reflex-driven insect brain and "phylogenetically evolves" into a complex,
object-tracking predator system.

### 1) Executive Summary

ProtoYolo is not a standard convolutional neural network. It is a hierarchical
spatiotemporal graph that evolves in phases.

- Input: high-temporal resolution video split into bio-channels (motion, contrast, spectral).
- Core mechanism: unsupervised Hebbian clustering (trace learning) plus hard-coded reflex arcs.
- Goal: replicate the survival loop: Detect -> Track -> Identify -> Act.

### 2) The "Genome": Base Architecture (The Insect Brain)

All subsequent evolutionary branches share this common core. This is the stem of
the phylogenetic tree.

#### A) The Sensorium (The Compound Eye)

Unlike standard YOLO (which sees RGB frames), ProtoYolo sees gradients.

- Channel 1: Luminance (L-Channel): high-pass filtered (edges only).
- Channel 2: Motion Energy (M-Channel): frame difference (t - t-1).
- Channel 3: Expansion (E-Channel): radial optical flow (looming).

#### B) The Primitive Reflexes (Hard-Coded Survival)

These are non-learning logic gates. They must exist before learning starts to
keep the agent alive.

- The Collider (Collision Avoidance):
  - Logic: If E-Channel (Expansion) > Threshold in Center Field -> Trigger Motor_Evasion.
- The Forager (Prey Capture):
  - Logic: If M-Channel (Motion) is "Small" AND "Fast" -> Trigger Motor_Approach.
- The Optomotor (Pathing/Stabilization):
  - Logic: If Global Flow is "Right-to-Left" -> Trigger Turn_Right (to stabilize view).

### 3) Phasic Evolution: The Roadmap

We simulate evolution by unlocking specific modules in phases. Each phase
represents a leap in complexity across animal phyla.

#### Phase I: The "Fly" (Diptera) - Reflexive Tracking

- Goal: Don't crash; chase small moving dots.
- YOLO status: region proposal only (no classification).
- Mechanism:
  - The system scans the M-Channel.
  - It draws bounding boxes around anything moving differently from the background.
  - Learning: none (purely hard-coded physics).
- Output: a stream of "Target" coordinates (x, y, v).

#### Phase II: The "Bee" (Hymenoptera) - Associative Learning

- Goal: Remember where the "Good Stuff" is (foraging/pathing).
- New module: color/feature association (Hebbian layer).
- Mechanism:
  - When the agent receives a reward (sugar), it captures the current visual texture
    (e.g., "Yellow + Radial Pattern").
  - Hebbian update: Texture_Neuron wires to Approach_Neuron.
  - Pathing: basic path integration (vector addition of turns).
    "I turned left, then right; Home is that way."
- YOLO status: proto-classifier. Can distinguish "Flower" (reward history)
  vs. "Non-Flower."

#### Phase III: The "Spider/Mantis" (Arachnida) - Object Permanence

- Goal: Ambush hunting. Track targets that stop moving.
- New module: trace memory (recurrent loops).
- Mechanism:
  - Problem: In Phase I, if a fly stops moving, it disappears from the M-Channel.
  - Solution: trace decay. If a target was at (x,y), keep neurons firing for
    2 seconds even if input stops.
- YOLO status: object tracking. The bounding box persists on stationary objects.

#### Phase IV: The "Vertebrate" Branch (Mammalia) - Semantic Identification

- Goal: Know what the target is (friend vs. foe).
- New module: deep cortex (the real YOLO).
- Mechanism:
  - The proto-boxes from Phase I/III are cropped and sent to a deep neural net.
  - Adaptive resolution: only high-saliency boxes are processed.
- YOLO status: full YOLO. Labels emerge ("Spider", "Mate", "Rock").

### 4) Engineering ProtoYolo (Implementation Plan)

Here is how to code the insect phase (Phase I and II) right now.

#### Step 1: The Bio-Preprocessor

Instead of feeding raw pixels to a neural net, feed it "Feature Maps."

def insect_retina(frame, prev_frame):
    # 1. Motion Channel (The Magno-cellular pathway)
    # Simple subtraction reveals moving things
    motion = cv2.absdiff(frame, prev_frame)

    # 2. Edge Channel (The Parvo-cellular pathway)
    # Laplacian filter mimics the retina's center-surround inhibition
    edges = cv2.Laplacian(frame, cv2.CV_64F)

    # 3. Saliency Map (The "Attention" Map)
    # Combine Motion + Edges. This is where the insect "looks."
    saliency = (motion * 0.8) + (edges * 0.2)

    return saliency

#### Step 2: The "Proto-Box" Generator (Unsupervised YOLO)

We do not predict classes. We predict "Objectness."

class ProtoObjectTracker:
    def __init__(self):
        self.object_files = []  # List of active tokens

    def update(self, saliency_map):
        # 1. Threshold the map to find "Blobs"
        _, thresh = cv2.threshold(saliency_map, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, ...)

        current_boxes = []
        for cnt in contours:
            # 2. Filter by "Physics" (Insect Logic)
            # Too big? It's a wall (background). Too small? Noise.
            area = cv2.contourArea(cnt)
            if 5 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                current_boxes.append((x, y, w, h))

        # 3. Match with Memory (Object Permanence)
        # Use IoU (Intersection over Union) to match new boxes to old files
        self.match_and_update_traces(current_boxes)

#### Step 3: Phylogenetic Branching (The Config)

You can control evolution via a config file.

# config_evolution.yaml

phylum: "Arthropoda"  # Options: Arthropoda, Chordata
development_stage: "Larva"  # Options: Larva, Pupa, Adult

vision:
  resolution: [64, 64]  # Insects have low spatial res
  temporal_res: 120     # But HIGH temporal res (FPS)
  color_channels: ["UV", "Green"]  # No Red channel for bees

brain:
  use_cortex: False     # Disable Deep YOLO (save compute)
  use_reflexes: True    # Enable looming/collision logic
  memory_decay: 0.5     # Short-term memory only (Fly) -> set to 0.99 for Spider

### 5) Deployment Strategy (The Agent)

To train this, do not use a dataset. Use a simulation environment.

- The Nursery: a simple 2D box with bouncing balls.
  - Task: avoid the big balls (looming) and catch the small balls (prey).
- The Garden: a complex 3D environment (Unity/PyGame).
  - Task: navigate to "Flowers" (Yellow cubes) while avoiding "Spiders" (Red cubes).
  - Trigger: forces the transition to Phase II (color association).
- The Forest: the full world.
  - Task: survive.
  - Trigger: forces the transition to Phase IV (semantic classification) because simple
    color rules fail (e.g., a green snake looks like grass).

### Next Step

Start by implementing the "Retina" (Step 1) to visualize what the
"Insect View" looks like compared to human vision.

---

### Insect Retina Script (reference implementation)

This script connects to a webcam or video file and produces the raw
bio-channels: motion, edges, and saliency.

You will need:

- `pip install opencv-python numpy`

#### 1) The code (`scripts/insect_retina.py`)

import cv2
import numpy as np

def insect_retina_stream(source=0):
    # Initialize video capture (0 = webcam, or use "video.mp4")
    cap = cv2.VideoCapture(source)

    # Read the first frame to initialize previous frame for motion detection
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return

    # Convert to grayscale (contrast, not color)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Blur slightly to remove camera noise (low-res eyes)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    print("Insect Retina Online. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # STEP 1: PRE-PROCESSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # STEP 2: M-CHANNEL (Motion Energy)
        motion_energy = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion_energy, 25, 255, cv2.THRESH_BINARY)

        # STEP 3: L-CHANNEL (Luminance/Edges)
        edges = cv2.Laplacian(gray_blurred, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)

        # STEP 4: SALIENCY MAP
        saliency = cv2.addWeighted(motion_mask, 0.8, edges, 0.2, 0)

        # Visualization
        display_shape = (320, 240)
        view_raw = cv2.resize(frame, display_shape)
        view_motion = cv2.resize(motion_mask, display_shape)
        view_saliency = cv2.resize(saliency, display_shape)

        view_motion_c = cv2.cvtColor(view_motion, cv2.COLOR_GRAY2BGR)
        view_saliency_c = cv2.cvtColor(view_saliency, cv2.COLOR_GRAY2BGR)

        combined_view = np.hstack((view_raw, view_motion_c, view_saliency_c))
        cv2.imshow("ProtoYolo Retina: [Human] vs [Motion] vs [Saliency]", combined_view)

        prev_gray = gray_blurred

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    insect_retina_stream()

#### 2) What you will see

When you run this, you will see three panels side-by-side:

- Left (Human View): normal video.
- Center (M-Channel): black screen that lights up only when things move.
- Right (Saliency Map): a mix of motion blobs and faint edges.

#### 3) Biological tuning

You can tune sensitivity by changing variables in the script:

- Temporal resolution: decrease blur from (21, 21) to (5, 5) for sharper movement.
- Attention balance: change addWeighted values.
  - 0.8 motion / 0.2 edge: predator mode (chase things).
  - 0.2 motion / 0.8 edge: navigator mode (avoid walls).

#### Next step

After the retina, implement Phase I (Collider): detect the center of mass of
saliency blobs and compute time-to-collision from looming.

---

### Phase I: The Collider (LGMD looming detector)

This module detects rapid expansion in the saliency map and triggers an evasion
command. It uses a simple time-to-collision heuristic: if the dominant blob grows
faster than a threshold for several frames, trigger EVADE.

#### Script: `scripts/insect_collider.py`

import cv2
import numpy as np

class LoomingDetector:
    def __init__(self, expansion_threshold=1.1):
        self.prev_area = 0.0
        self.expansion_thresh = float(expansion_threshold)
        self.threat_level = 0
        self.tracking_center = None

    def detect(self, saliency_map, debug_frame):
        """
        Scans the saliency map for looming threats.
        Returns: motor_command (str), updated_frame (image)
        """
        contours, _ = cv2.findContours(
            saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motor_command = "HOVER"
        color = (0, 255, 0)

        if not contours:
            self.prev_area = 0.0
            return motor_command, debug_frame

        largest_blob = max(contours, key=cv2.contourArea)
        current_area = float(cv2.contourArea(largest_blob))

        if current_area < 100.0:
            return motor_command, debug_frame

        x, y, w, h = cv2.boundingRect(largest_blob)
        center_x = x + w // 2

        if self.prev_area > 0.0:
            expansion_rate = current_area / self.prev_area
            if expansion_rate > self.expansion_thresh:
                self.threat_level += 1
            else:
                self.threat_level = max(0, self.threat_level - 1)
        else:
            expansion_rate = 1.0

        height, width = debug_frame.shape[:2]

        if self.threat_level > 3:
            color = (0, 0, 255)
            cv2.putText(
                debug_frame,
                "COLLISION IMMINENT!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )
            motor_command = "EVADE_RIGHT" if center_x < width // 2 else "EVADE_LEFT"

        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug_frame,
            f"Expansion: {expansion_rate:.2f}x",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        self.prev_area = current_area

        return motor_command, debug_frame

def run_collider_simulation(source=0):
    cap = cv2.VideoCapture(source)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return 1

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    brain = LoomingDetector(expansion_threshold=1.05)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        motion = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion, 30, 255, cv2.THRESH_BINARY)

        action, output_view = brain.detect(motion_mask, frame.copy())
        height, _ = output_view.shape[:2]
        cv2.putText(
            output_view,
            f"ACTION: {action}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("ProtoYolo: Phase I (Collider)", output_view)

        prev_gray = gray_blurred
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(run_collider_simulation())

#### How to test it

- Run the script.
- Stand back from the camera.
- Walk slowly forward: the box stays green.
- Move quickly toward the camera: the box turns red, and "EVADE_LEFT/RIGHT" appears.
- Move sideways: the box stays green.

#### Why this matters

This is a survival reflex. Before the agent learns what a "wall" or "predator" is,
it can already avoid collisions and survive long enough to start Hebbian learning.

---

### Phase II: The Forager (Prey Filter)

This module runs in parallel with the Collider. It filters for small, erratic motion
and triggers approach/locking behavior when a "prey spec" is detected.

#### High-level algorithm

- Input: saliency map (motion + edges).
- Blob analysis: find all moving contours.
- Filter:
  - Ignore large: area > threshold (handled by Collider).
  - Ignore noise: area < min_threshold.
  - Select small: min_threshold < area < max_threshold.
- Vector calculation: compute target centroid vs. screen center.
- Motor output: TURN_LEFT/RIGHT or FORWARD to minimize angle error.

#### Script: `scripts/insect_forager.py`

import cv2
import numpy as np

class InsectForager:
    def __init__(self, prey_min_size=10, prey_max_size=300):
        self.min_size = prey_min_size
        self.max_size = prey_max_size
        self.locked_target = None
        self.patience = 0

    def hunt(self, saliency_map, debug_frame):
        contours, _ = cv2.findContours(
            saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motor_command = "SEARCHING"
        height, width = debug_frame.shape[:2]
        center_screen = (width // 2, height // 2)

        potential_prey = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_size < area < self.max_size:
                potential_prey.append(cnt)

        if potential_prey:
            best_target = min(
                potential_prey,
                key=lambda cnt: np.linalg.norm(
                    np.array(cv2.boundingRect(cnt)[:2]) - np.array(center_screen)
                ),
            )
            x, y, w, h = cv2.boundingRect(best_target)
            target_center = (x + w // 2, y + h // 2)

            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.line(debug_frame, center_screen, target_center, (0, 255, 255), 1)

            error_x = target_center[0] - center_screen[0]
            if abs(error_x) < 50:
                motor_command = "FORWARD (CHASE)"
            elif error_x < 0:
                motor_command = "TURN_LEFT"
            else:
                motor_command = "TURN_RIGHT"

            self.locked_target = target_center
            self.patience = 10
        else:
            if self.patience > 0:
                self.patience -= 1
                motor_command = "SCANNING (LOST TARGET)"
            else:
                motor_command = "ROAMING"

        return motor_command, debug_frame

#### How to test the integrated agent

- Safety test: walk quickly toward the camera (Collider triggers EVADE).
- Hunting test: wave a finger/pen; a yellow box locks on and tries to center.
- Noise test: stay still; state should return to ROAMING.

#### Next step

Add Phase III (Object Permanence) so the prey target persists briefly even when motion stops.
