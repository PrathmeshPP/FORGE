import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

interface NavigationViewerProps {
  modelUrl: string | null;
  modelFile: File | null;
  onCameraUpdate: (pos: { x: number; y: number; z: number }, yaw: number) => void;
  onBoundsUpdate: (
    bounds: { minX: number; maxX: number; minZ: number; maxZ: number },
    footprintPoints: { x: number; z: number }[]
  ) => void;
  onLoadingChange: (loading: boolean, progress: number) => void;
  resetTrigger: number;
}

export const NavigationViewer: React.FC<NavigationViewerProps> = ({
  modelUrl,
  modelFile,
  onCameraUpdate,
  onBoundsUpdate,
  onLoadingChange,
  resetTrigger,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const currentModelRef = useRef<THREE.Group | null>(null);

  // Player position & orientation
  const playerPos = useRef<THREE.Vector3>(new THREE.Vector3(0, 1.2, 2.5));
  const playerYaw = useRef<number>(0);
  const playerPitch = useRef<number>(0);
  const velocity = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const keysPressed = useRef<{ [code: string]: boolean }>({});

  const isDragging = useRef<boolean>(false);
  const previousMousePosition = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const isPointerLocked = useRef<boolean>(false);

  const initialSpawn = useRef<{ pos: THREE.Vector3; yaw: number; pitch: number }>({
    pos: new THREE.Vector3(0, 1.2, 2.5),
    yaw: 0,
    pitch: 0,
  });

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // 1. Setup Three.js scene & renderer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 500);
    camera.rotation.order = 'YXZ';
    camera.position.copy(playerPos.current);
    cameraRef.current = camera;

    // Accurate realistic lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, 0.7);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
    keyLight.position.set(6, 12, 8);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.8);
    fillLight.position.set(-6, 6, -6);
    scene.add(fillLight);

    // Subtle floor reference grid
    const gridHelper = new THREE.GridHelper(20, 40, 0x38bdf8, 0x1e293b);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Resize handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Pointer lock listener
    const handlePointerLockChange = () => {
      isPointerLocked.current = document.pointerLockElement === renderer.domElement;
    };
    document.addEventListener('pointerlockchange', handlePointerLockChange);

    // Animation & Physics Loop
    let animationId: number;
    let lastTime = performance.now();

    const animate = () => {
      animationId = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      // Handle Keyboard Movement
      const keys = keysPressed.current;
      const moveSpeed = 3.2; // meters/sec
      const forwardDir = new THREE.Vector3(-Math.sin(playerYaw.current), 0, -Math.cos(playerYaw.current));
      const rightDir = new THREE.Vector3(Math.cos(playerYaw.current), 0, -Math.sin(playerYaw.current));

      const moveIntent = new THREE.Vector3(0, 0, 0);

      // W: Forward, S: Backward
      if (keys['KeyW'] || keys['ArrowUp']) moveIntent.add(forwardDir);
      if (keys['KeyS'] || keys['ArrowDown']) moveIntent.sub(forwardDir);

      // A: Left, D: Right
      if (keys['KeyA'] || keys['ArrowLeft']) moveIntent.sub(rightDir);
      if (keys['KeyD'] || keys['ArrowRight']) moveIntent.add(rightDir);

      // Upward in air: Space / Q
      if (keys['Space'] || keys['KeyQ']) moveIntent.y += 1.0;

      // Downward in air: Shift / C / E
      if (keys['ShiftLeft'] || keys['ShiftRight'] || keys['KeyC'] || keys['KeyE']) moveIntent.y -= 1.0;

      if (moveIntent.lengthSq() > 0.001) {
        moveIntent.normalize();
        const targetVel = moveIntent.multiplyScalar(moveSpeed);
        velocity.current.lerp(targetVel, Math.min(1.0, 10 * dt));
      } else {
        velocity.current.lerp(new THREE.Vector3(0, 0, 0), Math.min(1.0, 12 * dt));
      }

      playerPos.current.addScaledVector(velocity.current, dt);

      // Apply to Three.js camera
      camera.position.copy(playerPos.current);
      camera.rotation.y = playerYaw.current;
      camera.rotation.x = playerPitch.current;

      // Notify parent/minimap
      onCameraUpdate(
        { x: playerPos.current.x, y: playerPos.current.y, z: playerPos.current.z },
        playerYaw.current
      );

      renderer.render(scene, camera);
    };

    animationId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('pointerlockchange', handlePointerLockChange);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // 2. Keyboard & Mouse Input Listeners
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Prevent default scrolling on arrow keys and space
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        e.preventDefault();
      }
      keysPressed.current[e.code] = true;
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.code] = false;
    };

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        isDragging.current = true;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };

        // Double click or explicit click inside viewer to lock pointer
        if (e.detail >= 2 && rendererRef.current?.domElement) {
          rendererRef.current.domElement.requestPointerLock();
        }
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      const sensitivity = 0.0028;

      if (isPointerLocked.current) {
        playerYaw.current -= e.movementX * sensitivity;
        playerPitch.current = Math.max(
          -Math.PI / 2.05,
          Math.min(Math.PI / 2.05, playerPitch.current - e.movementY * sensitivity)
        );
      } else if (isDragging.current) {
        const deltaX = e.clientX - previousMousePosition.current.x;
        const deltaY = e.clientY - previousMousePosition.current.y;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };

        playerYaw.current -= deltaX * sensitivity;
        playerPitch.current = Math.max(
          -Math.PI / 2.05,
          Math.min(Math.PI / 2.05, playerPitch.current - deltaY * sensitivity)
        );
      }
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    const canvasDom = rendererRef.current?.domElement;
    if (canvasDom) {
      canvasDom.addEventListener('mousedown', handleMouseDown);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      if (canvasDom) {
        canvasDom.removeEventListener('mousedown', handleMouseDown);
      }
    };
  }, []);

  // 3. Reset Camera Hook
  const resetCamera = useCallback(() => {
    playerPos.current.copy(initialSpawn.current.pos);
    playerYaw.current = initialSpawn.current.yaw;
    playerPitch.current = initialSpawn.current.pitch;
    velocity.current.set(0, 0, 0);
  }, []);

  useEffect(() => {
    if (resetTrigger > 0) {
      resetCamera();
    }
  }, [resetTrigger, resetCamera]);

  // 4. Model Loading (Server URL or Local Upload File)
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    // Remove prior model
    if (currentModelRef.current) {
      scene.remove(currentModelRef.current);
      currentModelRef.current.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const m = child as THREE.Mesh;
          m.geometry?.dispose();
          if (Array.isArray(m.material)) {
            m.material.forEach((mat) => mat.dispose());
          } else {
            m.material?.dispose();
          }
        }
      });
      currentModelRef.current = null;
    }

    setErrorMessage(null);
    let objectUrlToRevoke: string | null = null;
    let targetUrl: string | null = null;

    if (modelFile) {
      objectUrlToRevoke = URL.createObjectURL(modelFile);
      targetUrl = objectUrlToRevoke;
    } else if (modelUrl) {
      targetUrl = modelUrl;
    }

    if (!targetUrl) return;

    onLoadingChange(true, 0);
    const loader = new GLTFLoader();

    loader.load(
      targetUrl,
      (gltf) => {
        onLoadingChange(false, 100);
        const model = gltf.scene;

        // Ensure double-sided rendering & standard materials
        model.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const m = child as THREE.Mesh;
            if (m.material) {
              if (Array.isArray(m.material)) {
                m.material.forEach((mat) => {
                  mat.side = THREE.DoubleSide;
                });
              } else {
                m.material.side = THREE.DoubleSide;
              }
            }
          }
        });

        scene.add(model);
        currentModelRef.current = model;

        // Compute Bounding Box
        const bbox = new THREE.Box3().setFromObject(model);
        const center = bbox.getCenter(new THREE.Vector3());

        const bounds = {
          minX: bbox.min.x,
          maxX: bbox.max.x,
          minZ: bbox.min.z,
          maxZ: bbox.max.z,
        };

        // Extract 2D footprint boundary points for Minimap
        const footprintPts = [
          { x: bbox.min.x, z: bbox.min.z },
          { x: bbox.max.x, z: bbox.min.z },
          { x: bbox.max.x, z: bbox.max.z },
          { x: bbox.min.x, z: bbox.max.z },
        ];
        onBoundsUpdate(bounds, footprintPts);

        // Place player inside or at natural spawn looking toward center
        const spawnY = Math.max(bbox.min.y + 0.3, center.y);
        const spawnZ = bbox.max.z - 0.2;
        const spawnPos = new THREE.Vector3(center.x, spawnY, spawnZ);

        initialSpawn.current = {
          pos: spawnPos.clone(),
          yaw: 0,
          pitch: 0,
        };

        playerPos.current.copy(spawnPos);
        playerYaw.current = 0;
        playerPitch.current = 0;
        velocity.current.set(0, 0, 0);

        if (objectUrlToRevoke) {
          URL.revokeObjectURL(objectUrlToRevoke);
        }
      },
      (xhr) => {
        if (xhr.lengthComputable && xhr.total > 0) {
          const pct = Math.round((xhr.loaded / xhr.total) * 100);
          onLoadingChange(true, pct);
        }
      },
      (err) => {
        console.error('Error loading 3D model:', err);
        setErrorMessage('Failed to load 3D model geometry. Please verify file integrity.');
        onLoadingChange(false, 0);
        if (objectUrlToRevoke) {
          URL.revokeObjectURL(objectUrlToRevoke);
        }
      }
    );
  }, [modelUrl, modelFile]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background: '#0a0f1d',
        cursor: isDragging.current ? 'grabbing' : 'grab',
      }}
    >
      {errorMessage && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'rgba(239, 68, 68, 0.9)',
            color: '#ffffff',
            padding: '16px 24px',
            borderRadius: '8px',
            fontFamily: 'system-ui, sans-serif',
            fontSize: '14px',
            zIndex: 30,
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          }}
        >
          {errorMessage}
        </div>
      )}
    </div>
  );
};
