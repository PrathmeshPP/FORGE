/**
 * F.O.R.G.E. - First-Person Indoor Navigation & Collision Engine.
 * 
 * Implements:
 * 1. PointerLock mouse-controlled camera (pitch & yaw with vertical clamping).
 * 2. WASD movement with Shift sprinting and wall-sliding physics.
 * 3. Spatial collision against obstacles (bed, desk, wardrobe, room walls).
 * 4. Floor height tracking and void/hole fall-protection clamping.
 * 5. Dynamic Euclidean proximity matching against original video capture trajectory.
 */

import * as THREE from 'three';
import walkthroughMetadata from '../data/indoor_walkthrough_metadata.json';

export interface TrajectoryPoint {
  index: number;
  position: [number, number, number];
  forward: [number, number, number];
  timestamp_sec: number;
  frame_url: string;
  filename: string;
}

export interface ArchitecturalRegion {
  id: string;
  name: string;
  type: string;
  bounds: [number, number, number, number]; // [x_min, z_min, x_max, z_max]
  height_range: [number, number];
  label: string;
}

export interface NavigationCallbacks {
  onPositionUpdate?: (pos: THREE.Vector3, yaw: number) => void;
  onClosestFrameChange?: (frame: TrajectoryPoint, distance: number) => void;
  onCollisionStatus?: (status: 'CLEAR' | 'SLIDING' | 'BOUNDARY') => void;
  onExitRequest?: () => void;
  onToggleMiniMap?: () => void;
  onTogglePointCloud?: () => void;
}

export class FirstPersonNavigationEngine {
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;
  private callbacks: NavigationCallbacks;

  // Player physics state
  private position: THREE.Vector3;
  private velocity: THREE.Vector3 = new THREE.Vector3();
  private yaw: number = 0;
  private pitch: number = 0;
  private isPointerLocked: boolean = false;
  private isSprinting: boolean = false;

  private moveState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
  };

  // Config parameters
  private readonly playerRadius = 0.18;
  private readonly eyeHeight = 0.58;
  private readonly floorY = -0.4867;
  private readonly walkSpeed = 1.15;
  private readonly sprintSpeed = 2.10;
  private readonly mouseSensitivity = 0.0022;

  // Collision boundaries
  private readonly xMin = -1.15;
  private readonly xMax = 0.95;
  private readonly zMin = -2.05;
  private readonly zMax = 1.35;

  private regions: ArchitecturalRegion[] = walkthroughMetadata.architectural_regions as ArchitecturalRegion[];
  private trajectory: TrajectoryPoint[] = walkthroughMetadata.camera_trajectory as TrajectoryPoint[];
  private lastClosestIndex: number = -1;

  // Bound event listeners
  private onKeyDownBound = this.onKeyDown.bind(this);
  private onKeyUpBound = this.onKeyUp.bind(this);
  private onMouseMoveBound = this.onMouseMove.bind(this);
  private onPointerLockChangeBound = this.onPointerLockChange.bind(this);

  constructor(
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement,
    callbacks: NavigationCallbacks = {}
  ) {
    this.camera = camera;
    this.domElement = domElement;
    this.callbacks = callbacks;

    // Spawn inside room at initial video capture location
    const spawn = walkthroughMetadata.player_physics.spawn_position;
    this.position = new THREE.Vector3(spawn[0], spawn[1], spawn[2]);
    this.yaw = Math.PI; // Face towards desk/interior
    this.pitch = 0.0;
  }

  public activate(): void {
    document.addEventListener('keydown', this.onKeyDownBound);
    document.addEventListener('keyup', this.onKeyUpBound);
    document.addEventListener('mousemove', this.onMouseMoveBound);
    document.addEventListener('pointerlockchange', this.onPointerLockChangeBound);

    this.requestPointerLock();
    this.syncCamera();
  }

  public deactivate(): void {
    document.removeEventListener('keydown', this.onKeyDownBound);
    document.removeEventListener('keyup', this.onKeyUpBound);
    document.removeEventListener('mousemove', this.onMouseMoveBound);
    document.removeEventListener('pointerlockchange', this.onPointerLockChangeBound);

    this.exitPointerLock();
  }

  public requestPointerLock(): void {
    if (this.domElement.requestPointerLock) {
      this.domElement.requestPointerLock();
    }
  }

  public exitPointerLock(): void {
    if (document.exitPointerLock) {
      document.exitPointerLock();
    }
  }

  private onPointerLockChange(): void {
    this.isPointerLocked = document.pointerLockElement === this.domElement;
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.isPointerLocked) return;

    this.yaw -= e.movementX * this.mouseSensitivity;
    this.pitch -= e.movementY * this.mouseSensitivity;

    // Clamp pitch between -85 and +85 degrees
    const maxPitch = (85 * Math.PI) / 180;
    this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));
  }

  private onKeyDown(e: KeyboardEvent): void {
    switch (e.code) {
      case 'KeyW':
        this.moveState.forward = true;
        break;
      case 'KeyS':
        this.moveState.backward = true;
        break;
      case 'KeyA':
        this.moveState.left = true;
        break;
      case 'KeyD':
        this.moveState.right = true;
        break;
      case 'ShiftLeft':
      case 'ShiftRight':
        this.isSprinting = true;
        break;
      case 'KeyM':
        if (this.callbacks.onToggleMiniMap) this.callbacks.onToggleMiniMap();
        break;
      case 'KeyP':
        if (this.callbacks.onTogglePointCloud) this.callbacks.onTogglePointCloud();
        break;
      case 'KeyG':
      case 'Escape':
        if (this.callbacks.onExitRequest) this.callbacks.onExitRequest();
        break;
    }
  }

  private onKeyUp(e: KeyboardEvent): void {
    switch (e.code) {
      case 'KeyW':
        this.moveState.forward = false;
        break;
      case 'KeyS':
        this.moveState.backward = false;
        break;
      case 'KeyA':
        this.moveState.left = false;
        break;
      case 'KeyD':
        this.moveState.right = false;
        break;
      case 'ShiftLeft':
      case 'ShiftRight':
        this.isSprinting = false;
        break;
    }
  }

  /**
   * Main physics & movement update loop (called on every requestAnimationFrame).
   */
  public update(deltaTimeSec: number): void {
    const dt = Math.min(deltaTimeSec, 0.05);

    // Calculate movement direction relative to camera yaw
    const forward = new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
    const right = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));

    const moveDir = new THREE.Vector3();
    if (this.moveState.forward) moveDir.add(forward);
    if (this.moveState.backward) moveDir.sub(forward);
    if (this.moveState.right) moveDir.add(right);
    if (this.moveState.left) moveDir.sub(right);

    const speed = this.isSprinting ? this.sprintSpeed : this.walkSpeed;

    if (moveDir.lengthSq() > 0.001) {
      moveDir.normalize();
      const targetVelocity = moveDir.clone().multiplyScalar(speed);
      this.velocity.lerp(targetVelocity, 0.25);
    } else {
      this.velocity.lerp(new THREE.Vector3(), 0.35);
    }

    // Attempted movement
    const displacement = this.velocity.clone().multiplyScalar(dt);
    if (displacement.lengthSq() > 1e-6) {
      this.applyCollisionAndMove(displacement);
    }

    // Keep camera anchored at human eye level above floor
    this.position.y = this.floorY + this.eyeHeight;

    this.syncCamera();

    // Trigger proximity tracking against original video frames
    this.updateProximityFrame();

    if (this.callbacks.onPositionUpdate) {
      this.callbacks.onPositionUpdate(this.position.clone(), this.yaw);
    }
  }

  /**
   * Resolves horizontal obstacle and boundary collision with sliding physics.
   */
  private applyCollisionAndMove(displacement: THREE.Vector3): void {
    let newX = this.position.x + displacement.x;
    let newZ = this.position.z + displacement.z;
    let status: 'CLEAR' | 'SLIDING' | 'BOUNDARY' = 'CLEAR';

    // 1. Room Boundary Clamping (prevents escaping outside scanned room)
    const clampedX = Math.max(this.xMin + this.playerRadius, Math.min(this.xMax - this.playerRadius, newX));
    const clampedZ = Math.max(this.zMin + this.playerRadius, Math.min(this.zMax - this.playerRadius, newZ));

    if (clampedX !== newX || clampedZ !== newZ) {
      status = 'BOUNDARY';
      newX = clampedX;
      newZ = clampedZ;
    }

    // 2. Obstacle Collision with Wall-Sliding (Bed, Desk, Wardrobe)
    for (const reg of this.regions) {
      if (reg.type !== 'obstacle') continue;

      const [ox1, oz1, ox2, oz2] = reg.bounds;
      const expandedX1 = ox1 - this.playerRadius;
      const expandedX2 = ox2 + this.playerRadius;
      const expandedZ1 = oz1 - this.playerRadius;
      const expandedZ2 = oz2 + this.playerRadius;

      // Check if new position penetrates obstacle bounding volume
      if (newX >= expandedX1 && newX <= expandedX2 && newZ >= expandedZ1 && newZ <= expandedZ2) {
        status = 'SLIDING';

        // Calculate overlap depths to find collision normal
        const dLeft = Math.abs(newX - expandedX1);
        const dRight = Math.abs(newX - expandedX2);
        const dFront = Math.abs(newZ - expandedZ1);
        const dBack = Math.abs(newZ - expandedZ2);

        const minOverlap = Math.min(dLeft, dRight, dFront, dBack);

        // Project out along smallest penetration axis (sliding response)
        if (minOverlap === dLeft) {
          newX = expandedX1;
        } else if (minOverlap === dRight) {
          newX = expandedX2;
        } else if (minOverlap === dFront) {
          newZ = expandedZ1;
        } else {
          newZ = expandedZ2;
        }
      }
    }

    this.position.x = newX;
    this.position.z = newZ;

    if (this.callbacks.onCollisionStatus) {
      this.callbacks.onCollisionStatus(status);
    }
  }

  /**
   * Finds the closest source video capture frame to current player position.
   */
  private updateProximityFrame(): void {
    if (!this.callbacks.onClosestFrameChange || this.trajectory.length === 0) return;

    let closestIdx = 0;
    let minDistance = Infinity;

    for (let i = 0; i < this.trajectory.length; i++) {
      const pt = this.trajectory[i];
      const dx = this.position.x - pt.position[0];
      const dy = this.position.y - pt.position[1];
      const dz = this.position.z - pt.position[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < minDistance) {
        minDistance = dist;
        closestIdx = i;
      }
    }

    if (closestIdx !== this.lastClosestIndex) {
      this.lastClosestIndex = closestIdx;
      this.callbacks.onClosestFrameChange(this.trajectory[closestIdx], minDistance);
    }
  }

  private syncCamera(): void {
    this.camera.position.copy(this.position);

    // Apply pitch & yaw to camera rotation
    const euler = new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(euler);
  }

  public getPosition(): THREE.Vector3 {
    return this.position.clone();
  }

  public getYaw(): number {
    return this.yaw;
  }

  public setPosition(pos: [number, number, number]): void {
    this.position.set(pos[0], pos[1], pos[2]);
    this.syncCamera();
  }
}

