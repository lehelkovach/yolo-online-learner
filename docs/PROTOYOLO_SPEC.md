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
