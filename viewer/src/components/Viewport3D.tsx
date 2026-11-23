import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { AlertTriangle, AlertCircle, Footprints } from 'lucide-react';
import type {
  SceneEntry,
  RepresentationType,
  ViewportConfig,
  BoundingBoxInfo,
  ActiveGeometryStats,
  CameraPresetAction,
} from '../types';

interface Viewport3DProps {
  scene: SceneEntry;
  representation: RepresentationType;
  config: ViewportConfig;
  onLoadingChange: (loading: boolean, progress: number) => void;
  onBoundingBoxChange?: (info: BoundingBoxInfo) => void;
  onStatsChange?: (stats: ActiveGeometryStats) => void;
  presetAction?: CameraPresetAction;
  isWalkthroughMode?: boolean;
  onToggleWalkthrough?: () => void;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  scene,
  representation,
  config,
  onLoadingChange,
  onBoundingBoxChange,
  onStatsChange,
  presetAction,
  isWalkthroughMode = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const modelGroupRef = useRef<THREE.Group | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);
  const axesHelperRef = useRef<THREE.AxesHelper | null>(null);

  // Lights refs for dynamic brightness scaling
  const ambientLightRef = useRef<THREE.AmbientLight | null>(null);
  const keyLightRef = useRef<THREE.DirectionalLight | null>(null);
  const fillLightRef = useRef<THREE.DirectionalLight | null>(null);
  const hemiLightRef = useRef<THREE.HemisphereLight | null>(null);

  // First-Person Walkthrough navigation state
  const isPointerLockedRef = useRef<boolean>(false);
  const [isPointerLocked, setIsPointerLocked] = useState<boolean>(false);
  const keysPressedRef = useRef<{ [key: string]: boolean }>({});
  const playerYawRef = useRef<number>(0);
  const playerPitchRef = useRef<number>(0);
  const playerPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const modelDimRef = useRef<{ radius: number; fitDist: number }>({ radius: 3, fitDist: 5 });

  const getBackgroundColor = (bg: string) => {
    switch (bg) {
      case 'black':
        return 0x05070a;
      case 'charcoal':
        return 0x111827;
      case 'light':
        return 0xe2e8f0;
      case 'slate':
      default:
        return 0x0f172a;
    }
  };

  const disposeObject = (obj: THREE.Object3D) => {
    obj.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((mat) => {
            if ((mat as any).map) (mat as any).map.dispose();
            mat.dispose();
          });
        } else if (mesh.material) {
          if ((mesh.material as any).map) (mesh.material as any).map.dispose();
          mesh.material.dispose();
        }
      }
    });
  };

  // 1. Initialize Three.js scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = config.brightness ?? 1.0;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(getBackgroundColor(config.background));
    sceneRef.current = threeScene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 1000);
    camera.position.set(...scene.initial_camera.position);
    cameraRef.current = camera;
    playerPosRef.current.copy(camera.position);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.0;
    controls.panSpeed = 0.8;
    controls.target.set(...scene.initial_camera.target);
    controlsRef.current = controls;

    // Lighting setup for indoor inspection
    const b = config.brightness ?? 1.0;
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2 * b);
    threeScene.add(ambientLight);
    ambientLightRef.current = ambientLight;

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.5 * b);
    keyLight.position.set(5, 10, 7);
    threeScene.add(keyLight);
    keyLightRef.current = keyLight;

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.8 * b);
    fillLight.position.set(-5, 5, -5);
    threeScene.add(fillLight);
    fillLightRef.current = fillLight;

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, 0.6 * b);
    threeScene.add(hemiLight);
    hemiLightRef.current = hemiLight;

    // Helpers
    const gridHelper = new THREE.GridHelper(10, 20, 0x38bdf8, 0x334155);
    gridHelper.position.y = -0.01;
    gridHelper.visible = config.showGrid;
    threeScene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    const axesHelper = new THREE.AxesHelper(1.5);
    axesHelper.visible = config.showAxes;
    threeScene.add(axesHelper);
    axesHelperRef.current = axesHelper;

    const modelGroup = new THREE.Group();
    threeScene.add(modelGroup);
    modelGroupRef.current = modelGroup;

    // First-person free-roam animation loop
    let animationFrameId: number;
    let lastTime = performance.now();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      if (isWalkthroughMode) {
        // Free-roam first-person movement (WASD + UP)
        const moveSpeed = 1.8 * dt; // 1.8 m/s
        const keys = keysPressedRef.current;
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.y = 0;
        forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

        const delta = new THREE.Vector3(0, 0, 0);

        if (keys['KeyW'] || keys['ArrowUp']) delta.addScaledVector(forward, moveSpeed);
        if (keys['KeyS'] || keys['ArrowDown']) delta.addScaledVector(forward, -moveSpeed);
        if (keys['KeyA'] || keys['ArrowLeft']) delta.addScaledVector(right, -moveSpeed);
        if (keys['KeyD'] || keys['ArrowRight']) delta.addScaledVector(right, moveSpeed);
        if (keys['KeyU']) delta.y += moveSpeed * 0.8;
        if (keys['KeyP']) delta.y -= moveSpeed * 0.8;

        if (delta.lengthSq() > 0) {
          camera.position.add(delta);
        }
      } else {
        // Orbit Controls update
        if (config.autoRotate && controlsRef.current) {
          controlsRef.current.autoRotate = true;
          controlsRef.current.autoRotateSpeed = 1.2;
        } else if (controlsRef.current) {
          controlsRef.current.autoRotate = false;
        }
        controls.update();
      }

      renderer.render(threeScene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // 2. Dynamic, Non-Destructive Brightness Scaling
  useEffect(() => {
    const b = config.brightness ?? 1.0;
    if (rendererRef.current) {
      rendererRef.current.toneMappingExposure = b;
    }
    if (ambientLightRef.current) ambientLightRef.current.intensity = 1.2 * b;
    if (keyLightRef.current) keyLightRef.current.intensity = 1.5 * b;
    if (fillLightRef.current) fillLightRef.current.intensity = 0.8 * b;
    if (hemiLightRef.current) hemiLightRef.current.intensity = 0.6 * b;
  }, [config.brightness]);

  // 3. Handle First-Person Mode switching and input listeners
  useEffect(() => {
    const controls = controlsRef.current;
    const camera = cameraRef.current;
    const container = containerRef.current;

    if (isWalkthroughMode) {
      if (controls) controls.enabled = false;

      const onKeyDown = (e: KeyboardEvent) => {
        keysPressedRef.current[e.code] = true;
      };
      const onKeyUp = (e: KeyboardEvent) => {
        keysPressedRef.current[e.code] = false;
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isPointerLockedRef.current || !camera) return;
        const sensitivity = 0.0022;
        playerYawRef.current -= e.movementX * sensitivity;
        playerPitchRef.current = Math.max(
          -Math.PI / 2.2,
          Math.min(Math.PI / 2.2, playerPitchRef.current - e.movementY * sensitivity)
        );

        const euler = new THREE.Euler(0, 0, 0, 'YXZ');
        euler.x = playerPitchRef.current;
        euler.y = playerYawRef.current;
        camera.quaternion.setFromEuler(euler);
      };

      const onClickCanvas = () => {
        if (container && !document.pointerLockElement) {
          container.requestPointerLock?.();
        }
      };

      const onPointerLockChange = () => {
        const locked = document.pointerLockElement === container;
        isPointerLockedRef.current = locked;
        setIsPointerLocked(locked);
      };

      window.addEventListener('keydown', onKeyDown);
      window.addEventListener('keyup', onKeyUp);
      window.addEventListener('mousemove', onMouseMove);
      container?.addEventListener('click', onClickCanvas);
      document.addEventListener('pointerlockchange', onPointerLockChange);

      return () => {
        window.removeEventListener('keydown', onKeyDown);
        window.removeEventListener('keyup', onKeyUp);
        window.removeEventListener('mousemove', onMouseMove);
        container?.removeEventListener('click', onClickCanvas);
        document.removeEventListener('pointerlockchange', onPointerLockChange);
        if (document.pointerLockElement === container) {
          document.exitPointerLock?.();
        }
        keysPressedRef.current = {};
      };
    } else {
      if (controls) controls.enabled = true;
      if (document.pointerLockElement === container) {
        document.exitPointerLock?.();
      }
      isPointerLockedRef.current = false;
      setIsPointerLocked(false);
    }
  }, [isWalkthroughMode]);

  // 4. Handle Preset Camera Actions
  useEffect(() => {
    if (!presetAction || !cameraRef.current || !controlsRef.current) return;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const d = modelDimRef.current.fitDist || 3.5;

    switch (presetAction.type) {
      case 'top':
        camera.position.set(0, d * 1.5, 0.001);
        controls.target.set(0, 0, 0);
        break;
      case 'front':
        camera.position.set(0, 0, d * 1.5);
        controls.target.set(0, 0, 0);
        break;
      case 'side':
        camera.position.set(d * 1.5, 0, 0);
        controls.target.set(0, 0, 0);
        break;
      case 'iso':
        camera.position.set(d * 0.8, d * 0.8, d * 0.8);
        controls.target.set(0, 0, 0);
        break;
      case 'reset':
        if (scene.initial_camera) {
          camera.position.set(...scene.initial_camera.position);
          controls.target.set(...scene.initial_camera.target);
        } else {
          camera.position.set(0, d * 0.75, d * 0.85);
          controls.target.set(0, 0, 0);
        }
        break;
    }
    controls.update();
  }, [presetAction, scene]);

  // 5. Update helpers and background on config change
  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = config.showGrid;
    if (axesHelperRef.current) axesHelperRef.current.visible = config.showAxes;
    if (sceneRef.current) {
      sceneRef.current.background = new THREE.Color(getBackgroundColor(config.background));
    }
    if (modelGroupRef.current) {
      modelGroupRef.current.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => {
              (m as any).wireframe = config.wireframe;
            });
          } else if (mesh.material) {
            (mesh.material as any).wireframe = config.wireframe;
          }
        }
      });
    }
  }, [config.showGrid, config.showAxes, config.background, config.wireframe]);

  // 6. Load Model Asset (GLB, OBJ, PLY)
  const loadModel = useCallback(() => {
    const modelGroup = modelGroupRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const container = containerRef.current;
    if (!modelGroup || !camera || !controls) return;

    // Thorough GPU memory cleanup before loading next model
    while (modelGroup.children.length > 0) {
      const child = modelGroup.children[0];
      modelGroup.remove(child);
      disposeObject(child);
    }

    setErrorMessage(null);
    onLoadingChange(true, 0);

    const repConfig =
      scene.representations[representation] ||
      scene.representations.glb ||
      scene.representations.textured_obj ||
      Object.values(scene.representations)[0];

    const setupLoadedObject = (object: THREE.Object3D, format: 'GLB' | 'OBJ' | 'PLY', sourceFileName: string) => {
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      const radius = sphere.radius || Math.max(size.x, size.y, size.z) / 2 || 1;

      if (onBoundingBoxChange) {
        onBoundingBoxChange({
          min: box.min.toArray() as [number, number, number],
          max: box.max.toArray() as [number, number, number],
          size: size.toArray() as [number, number, number],
          center: center.toArray() as [number, number, number],
        });
      }

      // Center container group so camera orbits around center
      object.position.set(-center.x, -center.y, -center.z);

      object.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => {
              (m as any).wireframe = config.wireframe;
              m.side = THREE.DoubleSide;
            });
          } else if (mesh.material) {
            (mesh.material as any).wireframe = config.wireframe;
            mesh.material.side = THREE.DoubleSide;
          }
        }
      });

      modelGroup.add(object);

      const width = container?.clientWidth || 800;
      const height = container?.clientHeight || 600;
      const aspect = width / height;
      const fovRad = (camera.fov * Math.PI) / 180;
      const vDist = radius / Math.sin(fovRad / 2);
      const hDist = radius / Math.sin(Math.atan(Math.tan(fovRad / 2) * aspect));
      const fitDist = Math.max(vDist, hDist) * 1.15;

      modelDimRef.current = { radius, fitDist };

      camera.near = Math.max(0.01, fitDist / 150);
      camera.far = Math.max(1000, fitDist * 35);
      camera.updateProjectionMatrix();

      if (scene.initial_camera) {
        camera.position.set(...scene.initial_camera.position);
        controls.target.set(...scene.initial_camera.target);
      } else {
        camera.position.set(0, fitDist * 0.75, fitDist * 0.85);
        controls.target.set(0, 0, 0);
      }
      controls.update();

      let renderedPositions = 0;
      let triangleCount = 0;

      object.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const geom = (child as THREE.Mesh).geometry;
          if (geom) {
            renderedPositions += geom.attributes.position?.count || 0;
            if (geom.index) {
              triangleCount += Math.round(geom.index.count / 3);
            } else {
              triangleCount += Math.round((geom.attributes.position?.count || 0) / 3);
            }
          }
        }
      });

      const stats: ActiveGeometryStats = {
        uniqueVertices: scene.vertex_count || (format === 'PLY' ? renderedPositions : null),
        renderedPositions: renderedPositions,
        triangles: triangleCount,
        sourceFileName: sourceFileName,
        format: format as any,
      };

      if (onStatsChange) {
        onStatsChange(stats);
      }
      onLoadingChange(false, 100);
    };

    // A. Load GLB (Primary Photographic Reconstruction Format)
    if (repConfig?.type === 'glb' || repConfig?.glb_url) {
      const glbUrl = repConfig.glb_url || (repConfig as any).url;
      const sourceFileName = glbUrl.split('/').pop() || 'room_photographic_model.glb';
      const gltfLoader = new GLTFLoader();

      gltfLoader.load(
        glbUrl,
        (gltf) => {
          setupLoadedObject(gltf.scene, 'GLB', sourceFileName);
        },
        (xhr) => {
          if (xhr.total > 0) {
            onLoadingChange(true, Math.round((xhr.loaded / xhr.total) * 100));
          }
        },
        (_err) => {
          setErrorMessage(`Failed to load GLB model from ${glbUrl}. Verify file exists on host.`);
          onLoadingChange(false, 0);
        }
      );
    }
    // B. Load OBJ + MTL
    else if (repConfig?.type === 'obj' && repConfig.obj_url) {
      const objUrl = repConfig.obj_url;
      const mtlUrl = repConfig.mtl_url;
      const sourceFileName = objUrl.split('/').pop() || 'textured_mesh.obj';
      const objLoader = new OBJLoader();

      if (mtlUrl) {
        const mtlLoader = new MTLLoader();
        const basePath = mtlUrl.substring(0, mtlUrl.lastIndexOf('/') + 1);
        mtlLoader.setPath(basePath);
        const mtlFileName = mtlUrl.substring(mtlUrl.lastIndexOf('/') + 1);

        mtlLoader.load(
          mtlFileName,
          (materials) => {
            materials.preload();
            objLoader.setMaterials(materials);
            objLoader.load(
              objUrl,
              (obj) => setupLoadedObject(obj, 'OBJ', sourceFileName),
              undefined,
              () => {
                setErrorMessage('Failed to load OBJ geometry. Ensure asset path is valid.');
                onLoadingChange(false, 0);
              }
            );
          },
          undefined,
          () => {
            // Fallback: load OBJ without materials if MTL fails
            objLoader.load(
              objUrl,
              (obj) => setupLoadedObject(obj, 'OBJ', sourceFileName),
              undefined,
              () => {
                setErrorMessage('Failed to load OBJ geometry file.');
                onLoadingChange(false, 0);
              }
            );
          }
        );
      } else {
        objLoader.load(
          objUrl,
          (obj) => setupLoadedObject(obj, 'OBJ', sourceFileName),
          undefined,
          () => {
            setErrorMessage('Failed to load OBJ file.');
            onLoadingChange(false, 0);
          }
        );
      }
    }
    // C. Load PLY
    else if (repConfig?.type === 'ply' && repConfig.ply_url) {
      const plyUrl = repConfig.ply_url;
      const sourceFileName = plyUrl.split('/').pop() || 'surface_mesh.ply';
      const plyLoader = new PLYLoader();

      plyLoader.load(
        plyUrl,
        (geometry) => {
          geometry.computeVertexNormals();

          let material: THREE.Material;
          if (geometry.hasAttribute('color')) {
            material = new THREE.MeshStandardMaterial({
              vertexColors: true,
              roughness: 0.8,
              metalness: 0.1,
              side: THREE.DoubleSide,
            });
          } else {
            material = new THREE.MeshStandardMaterial({
              color: 0x94a3b8,
              roughness: 0.7,
              metalness: 0.1,
              side: THREE.DoubleSide,
            });
          }

          const mesh = new THREE.Mesh(geometry, material);
          setupLoadedObject(mesh, 'PLY', sourceFileName);
        },
        undefined,
        () => {
          setErrorMessage('Failed to load Stanford PLY format geometry.');
          onLoadingChange(false, 0);
        }
      );
    } else {
      setErrorMessage('No valid 3D representation found for this scene.');
      onLoadingChange(false, 0);
    }
  }, [scene, representation, onLoadingChange, onBoundingBoxChange, onStatsChange, config.wireframe]);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  return (
    <div className="viewport-wrapper">
      <div ref={containerRef} className="three-canvas-container" />

      {/* SAMPLE ASSET WATERMARK BANNER */}
      {scene.is_sample_asset && (
        <div className="sample-watermark-overlay">
          <AlertTriangle size={15} className="watermark-icon" />
          <span>{scene.sample_watermark_text || 'SAMPLE TEST FIXTURE — NOT AN ACTUAL VIDEO RECONSTRUCTION'}</span>
        </div>
      )}

      {/* WALKTHROUGH MODE OVERLAY INSTRUCTIONS */}
      {isWalkthroughMode && (
        <div className="walkthrough-overlay-hud">
          <div className="hud-badge">
            <Footprints size={14} className="text-emerald" />
            <span>First-Person Walkthrough Active</span>
          </div>
          <div className="hud-controls-hint">
            <span><strong>W/S</strong>: Move Forward/Back &bull; <strong>A/D</strong>: Strafe &bull; <strong>U/P</strong>: Up/Down</span>
            <span className="hud-subhint">
              {isPointerLocked ? (
                <span>&bull; Mouse Look Active (Press ESC to release cursor)</span>
              ) : (
                <span>&bull; Click viewport to engage mouse look</span>
              )}
            </span>
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="viewport-error-overlay">
          <AlertCircle size={20} />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
