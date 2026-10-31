import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type {
  SceneEntry,
  RepresentationType,
  ViewportConfig,
  BoundingBoxInfo,
  ActiveGeometryStats,
  CameraPresetAction,
} from '../types';
import { GeometryStatsDisplay } from './GeometryStatsDisplay';
import { FirstPersonNavigationEngine } from '../navigation/FirstPersonNavigationEngine';
import type { TrajectoryPoint } from '../navigation/FirstPersonNavigationEngine';
import { MiniMapOverlay } from './MiniMapOverlay';
import { SourceFramePreview } from './SourceFramePreview';
import { WalkthroughHUD } from './WalkthroughHUD';

interface Viewport3DProps {
  scene: SceneEntry;
  representation: RepresentationType;
  config: ViewportConfig;
  onLoadingChange: (loading: boolean, progress: number) => void;
  onBoundingBoxChange?: (info: BoundingBoxInfo) => void;
  onStatsChange?: (stats: ActiveGeometryStats) => void;
  syncCameraState?: {
    position: [number, number, number];
    target: [number, number, number];
  };
  onCameraUpdate?: (pos: [number, number, number], target: [number, number, number]) => void;
  customTitle?: string;
  onFallbackToPly?: () => void;
  presetAction?: CameraPresetAction;
  onSelectRepresentation?: (rep: RepresentationType) => void;
  isWalkthroughMode?: boolean;
  onToggleWalkthrough?: (active: boolean) => void;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  scene,
  representation,
  config,
  onLoadingChange,
  onBoundingBoxChange,
  onStatsChange,
  syncCameraState,
  onCameraUpdate,
  customTitle,
  onFallbackToPly,
  presetAction,
  onSelectRepresentation,
  isWalkthroughMode: externalWalkthroughMode,
  onToggleWalkthrough,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const modelGroupRef = useRef<THREE.Group | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);
  const axesHelperRef = useRef<THREE.AxesHelper | null>(null);

  // Walkthrough State
  const [internalWalkthrough, setInternalWalkthrough] = useState<boolean>(false);
  const isWalkthrough = externalWalkthroughMode !== undefined ? externalWalkthroughMode : internalWalkthrough;
  const isWalkthroughRef = useRef<boolean>(isWalkthrough);
  isWalkthroughRef.current = isWalkthrough;

  const navEngineRef = useRef<FirstPersonNavigationEngine | null>(null);
  const clockRef = useRef<THREE.Clock>(new THREE.Clock());

  const [playerPos, setPlayerPos] = useState<[number, number, number]>([0.12, 0.09, -1.27]);
  const [playerYaw, setPlayerYaw] = useState<number>(Math.PI);
  const [collisionStatus, setCollisionStatus] = useState<'CLEAR' | 'SLIDING' | 'BOUNDARY'>('CLEAR');
  const [fps, setFps] = useState<number>(60);
  const [currentFrame, setCurrentFrame] = useState<TrajectoryPoint | null>(null);
  const [frameDistance, setFrameDistance] = useState<number>(0);
  const [isMiniMapOpen, setIsMiniMapOpen] = useState<boolean>(true);
  const [isPointerLocked, setIsPointerLocked] = useState<boolean>(false);

  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeStats, setActiveStats] = useState<ActiveGeometryStats | null>(null);
  const modelDimRef = useRef<{ radius: number; fitDist: number }>({ radius: 3, fitDist: 5 });

  const isIndoor = scene.id.startsWith('indoor_') || scene.category === 'Indoor Architecture';

  // Background color mapping
  const getBackgroundColor = (bg: string) => {
    switch (bg) {
      case 'black':
        return 0x000000;
      case 'charcoal':
        return 0x111827;
      case 'light':
        return 0xe2e8f0;
      case 'slate':
      default:
        return 0x0f172a;
    }
  };

  // Safe resource disposal helper
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

  // Initialize Three.js scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Scene
    const threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(getBackgroundColor(config.background));
    sceneRef.current = threeScene;

    // Camera
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.05, 1000);
    camera.position.set(...scene.initial_camera.position);
    cameraRef.current = camera;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.0;
    controls.panSpeed = 0.8;
    controls.target.set(...scene.initial_camera.target);
    controlsRef.current = controls;

    // Balanced Lighting for Architectural Interiors
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    threeScene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(0, 5, 2);
    threeScene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.55);
    fillLight.position.set(-2, 2, -2);
    threeScene.add(fillLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, 0.65);
    threeScene.add(hemiLight);

    // Helpers
    const gridHelper = new THREE.GridHelper(10, 20, 0x38bdf8, 0x334155);
    gridHelper.position.y = isIndoor ? -0.487 : -0.01;
    gridHelper.visible = config.showGrid && !isWalkthroughRef.current;
    threeScene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    const axesHelper = new THREE.AxesHelper(1.5);
    axesHelper.visible = config.showAxes && !isWalkthroughRef.current;
    threeScene.add(axesHelper);
    axesHelperRef.current = axesHelper;

    // Model Group
    const modelGroup = new THREE.Group();
    threeScene.add(modelGroup);
    modelGroupRef.current = modelGroup;

    // Camera change listener for syncing in inspection mode
    controls.addEventListener('change', () => {
      if (!isWalkthroughRef.current && onCameraUpdate && cameraRef.current && controlsRef.current) {
        const pos = cameraRef.current.position.toArray() as [number, number, number];
        const target = controlsRef.current.target.toArray() as [number, number, number];
        onCameraUpdate(pos, target);
      }
    });

    // Pointer lock change listener
    const onPLChange = () => {
      setIsPointerLocked(document.pointerLockElement === renderer.domElement);
    };
    document.addEventListener('pointerlockchange', onPLChange);

    // Animation Loop
    let animationFrameId: number;
    clockRef.current.start();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (isWalkthroughRef.current && navEngineRef.current) {
        const delta = clockRef.current.getDelta();
        navEngineRef.current.update(delta);

        // FPS measurement
        frameCountRef.current++;
        const now = performance.now();
        if (now - lastFpsTimeRef.current >= 500) {
          const measured = Math.round((frameCountRef.current * 1000) / (now - lastFpsTimeRef.current));
          setFps(measured);
          frameCountRef.current = 0;
          lastFpsTimeRef.current = now;
        }
      } else {
        if (config.autoRotate && controlsRef.current) {
          controlsRef.current.autoRotate = true;
          controlsRef.current.autoRotateSpeed = 1.2;
        } else if (controlsRef.current) {
          controlsRef.current.autoRotate = false;
        }
        if (controlsRef.current && controlsRef.current.enabled) {
          controlsRef.current.update();
        }
      }

      renderer.render(threeScene, camera);
    };
    animate();

    // Resize Handler
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
      document.removeEventListener('pointerlockchange', onPLChange);
      if (navEngineRef.current) {
        navEngineRef.current.deactivate();
      }
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Enter / Exit Walkthrough Handler
  const handleEnterWalkthrough = useCallback(() => {
    if (!cameraRef.current || !rendererRef.current || !controlsRef.current) return;

    controlsRef.current.enabled = false;
    cameraRef.current.near = 0.05;
    cameraRef.current.far = 100;
    cameraRef.current.updateProjectionMatrix();

    if (gridHelperRef.current) gridHelperRef.current.visible = false;
    if (axesHelperRef.current) axesHelperRef.current.visible = false;

    if (!navEngineRef.current) {
      navEngineRef.current = new FirstPersonNavigationEngine(
        cameraRef.current,
        rendererRef.current.domElement,
        {
          onPositionUpdate: (pos, yaw) => {
            setPlayerPos([pos.x, pos.y, pos.z]);
            setPlayerYaw(yaw);
          },
          onClosestFrameChange: (frame, dist) => {
            setCurrentFrame(frame);
            setFrameDistance(dist);
          },
          onCollisionStatus: (status) => {
            setCollisionStatus(status);
          },
          onExitRequest: () => {
            handleExitWalkthrough();
          },
          onToggleMiniMap: () => {
            setIsMiniMapOpen((prev) => !prev);
          },
          onTogglePointCloud: () => {
            if (onSelectRepresentation) {
              const nextRep: RepresentationType = representation === 'photographic_ply' ? 'gltf_glb' : 'photographic_ply';
              onSelectRepresentation(nextRep);
            }
          },
        }
      );
    }

    navEngineRef.current.activate();

    if (onToggleWalkthrough) {
      onToggleWalkthrough(true);
    } else {
      setInternalWalkthrough(true);
    }
  }, [representation, onSelectRepresentation, onToggleWalkthrough]);

  const handleExitWalkthrough = useCallback(() => {
    console.log('[Viewport3D] handleExitWalkthrough called');
    if (!controlsRef.current || !cameraRef.current) return;

    if (navEngineRef.current) {
      navEngineRef.current.deactivate();
      navEngineRef.current = null;
    }

    controlsRef.current.enabled = true;

    // Reset to inspection view
    if (scene.initial_camera) {
      cameraRef.current.position.set(...scene.initial_camera.position);
      controlsRef.current.target.set(...scene.initial_camera.target);
    } else {
      cameraRef.current.position.set(0, 0.75, 1.2);
      controlsRef.current.target.set(0, 0, 0);
    }

    cameraRef.current.near = isIndoor ? 0.05 : 0.1;
    cameraRef.current.updateProjectionMatrix();
    controlsRef.current.update();

    if (gridHelperRef.current) gridHelperRef.current.visible = config.showGrid;
    if (axesHelperRef.current) axesHelperRef.current.visible = config.showAxes;

    if (onToggleWalkthrough) {
      console.log('[Viewport3D] Calling onToggleWalkthrough(false)');
      onToggleWalkthrough(false);
    } else {
      setInternalWalkthrough(false);
    }
  }, [scene, isIndoor, config, onToggleWalkthrough]);

  // Sync external walkthrough state if provided
  useEffect(() => {
    if (externalWalkthroughMode !== undefined) {
      if (externalWalkthroughMode && !navEngineRef.current) {
        handleEnterWalkthrough();
      } else if (!externalWalkthroughMode && navEngineRef.current) {
        handleExitWalkthrough();
      }
    }
  }, [externalWalkthroughMode, handleEnterWalkthrough, handleExitWalkthrough]);

  // Sync camera state if provided externally
  useEffect(() => {
    if (syncCameraState && cameraRef.current && controlsRef.current && !isWalkthroughRef.current) {
      cameraRef.current.position.set(...syncCameraState.position);
      controlsRef.current.target.set(...syncCameraState.target);
      controlsRef.current.update();
    }
  }, [syncCameraState]);

  // Handle Preset Camera Actions
  useEffect(() => {
    if (isWalkthroughRef.current || !presetAction || !cameraRef.current || !controlsRef.current) return;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const d = modelDimRef.current.fitDist || 3.5;

    switch (presetAction.type) {
      case 'top':
        camera.position.set(0, isIndoor ? 1.8 : d * 1.5, 0.001);
        controls.target.set(0, isIndoor ? -0.2 : 0, 0);
        break;
      case 'front':
        camera.position.set(0, isIndoor ? 0.2 : 0, isIndoor ? 1.5 : d * 1.5);
        controls.target.set(0, 0, 0);
        break;
      case 'side':
        camera.position.set(isIndoor ? 1.5 : d * 1.5, isIndoor ? 0.2 : 0, 0);
        controls.target.set(0, 0, 0);
        break;
      case 'iso':
        camera.position.set(isIndoor ? 1.0 : d * 0.8, isIndoor ? 0.9 : d * 0.8, isIndoor ? 1.2 : d * 0.8);
        controls.target.set(0, isIndoor ? -0.1 : 0, 0);
        break;
      case 'reset':
      case 'overview':
        if (scene.initial_camera) {
          camera.position.set(...scene.initial_camera.position);
          controls.target.set(...scene.initial_camera.target);
        } else {
          camera.position.set(0, isIndoor ? 0.5 : d * 0.75, isIndoor ? 1.5 : d * 0.85);
          controls.target.set(0, 0, 0);
        }
        break;
      default:
        break;
    }
    controls.update();
  }, [presetAction, scene, isIndoor]);

  // Update helpers and background on config change
  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = config.showGrid && !isWalkthroughRef.current;
    if (axesHelperRef.current) axesHelperRef.current.visible = config.showAxes && !isWalkthroughRef.current;
    if (sceneRef.current) {
      sceneRef.current.background = new THREE.Color(getBackgroundColor(config.background));
    }
    // Update wireframe state on current model
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
  }, [config]);

  // Load Model Asset
  const loadModel = useCallback(() => {
    const modelGroup = modelGroupRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const container = containerRef.current;
    if (!modelGroup || !camera || !controls) return;

    // Clear previous model
    while (modelGroup.children.length > 0) {
      const child = modelGroup.children[0];
      modelGroup.remove(child);
      disposeObject(child);
    }

    setErrorMessage(null);
    onLoadingChange(true, 0);

    const repConfig = scene.representations[representation] || scene.representations.textured_obj || scene.representations.gltf_glb;
    const onProgress = (xhr: ProgressEvent) => {
      if (xhr.lengthComputable && xhr.total > 0) {
        const pct = Math.round((xhr.loaded / xhr.total) * 100);
        onLoadingChange(true, pct);
      }
    };

    const setupLoadedObject = (object: THREE.Object3D, format: 'OBJ' | 'PLY' | 'GLB', sourceFileName: string) => {
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

      // For indoor models, coordinates are canonically aligned and centered at origin!
      if (isIndoor) {
        object.position.set(0, 0, 0);
      } else {
        object.position.set(-center.x, -center.y, -center.z);
      }

      // Material setup: Double-sided, proper PBR roughness, sRGB
      object.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          const geom = mesh.geometry;
          const hasColors = geom && !!geom.attributes.color;

          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => {
              (m as any).wireframe = config.wireframe;
              m.side = THREE.DoubleSide;
              if (hasColors) {
                (m as any).vertexColors = true;
                m.needsUpdate = true;
              }
              if ((m as any).map) {
                (m as any).map.colorSpace = THREE.SRGBColorSpace;
              }
            });
          } else if (mesh.material) {
            (mesh.material as any).wireframe = config.wireframe;
            mesh.material.side = THREE.DoubleSide;
            if (hasColors) {
              (mesh.material as any).vertexColors = true;
              mesh.material.needsUpdate = true;
            }
            if ((mesh.material as any).map) {
              (mesh.material as any).map.colorSpace = THREE.SRGBColorSpace;
            }
          }
        }
      });

      modelGroup.add(object);

      // Adaptive clipping planes
      const width = container?.clientWidth || 800;
      const height = container?.clientHeight || 600;
      const aspect = width / height;
      const fovRad = (camera.fov * Math.PI) / 180;
      const vDist = radius / Math.sin(fovRad / 2);
      const hDist = radius / Math.sin(Math.atan(Math.tan(fovRad / 2) * aspect));
      const fitDist = Math.max(vDist, hDist) * 1.15;

      modelDimRef.current = { radius, fitDist };

      camera.near = isIndoor ? 0.05 : Math.max(0.01, fitDist / 150);
      camera.far = isIndoor ? 100 : Math.max(1000, fitDist * 35);
      camera.updateProjectionMatrix();

      if (!isWalkthroughRef.current) {
        if (scene.initial_camera) {
          camera.position.set(...scene.initial_camera.position);
          controls.target.set(...scene.initial_camera.target);
        } else {
          camera.position.set(0, fitDist * 0.75, fitDist * 0.85);
          controls.target.set(0, 0, 0);
        }
        controls.update();
      }

      let renderedPositions = 0;
      let triangleCount = 0;

      object.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const geom = (child as THREE.Mesh).geometry;
          if (geom) {
            renderedPositions += geom.attributes.position?.count || 0;
            if (geom.index) {
              triangleCount += Math.round(geom.index.count / 3);
            } else if (geom.attributes.position) {
              triangleCount += Math.round(geom.attributes.position.count / 3);
            }
          }
        }
      });

      const stats: ActiveGeometryStats = {
        uniqueVertices: scene.vertex_count || (format === 'PLY' ? renderedPositions : null),
        renderedPositions: renderedPositions,
        triangles: triangleCount,
        sourceFileName: sourceFileName,
        format: format,
      };

      setActiveStats(stats);
      if (onStatsChange) {
        onStatsChange(stats);
      }

      onLoadingChange(false, 100);
    };

    // Branch by loader type
    if (repConfig.type === 'obj' && repConfig.obj_url) {
      const mtlUrl = repConfig.mtl_url;
      const objUrl = repConfig.obj_url;
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
              onProgress,
              (err) => {
                console.error('OBJ load error:', err);
                setErrorMessage('Failed to load textured OBJ model');
                onLoadingChange(false, 0);
              }
            );
          },
          onProgress,
          (err) => {
            console.warn('MTL load error, falling back to untextured OBJ:', err);
            objLoader.load(
              objUrl,
              (obj) => setupLoadedObject(obj, 'OBJ', sourceFileName),
              onProgress,
              () => {
                setErrorMessage('Failed to load OBJ model');
                onLoadingChange(false, 0);
              }
            );
          }
        );
      } else {
        objLoader.load(
          objUrl,
          (obj) => setupLoadedObject(obj, 'OBJ', sourceFileName),
          onProgress,
          () => {
            setErrorMessage('Failed to load OBJ model');
            onLoadingChange(false, 0);
          }
        );
      }
    } else if (repConfig.type === 'ply' && repConfig.ply_url) {
      const plyUrl = repConfig.ply_url;
      const sourceFileName = plyUrl.split('/').pop() || 'surface_mesh.ply';
      const plyLoader = new PLYLoader();
      plyLoader.load(
        plyUrl,
        (geometry) => {
          geometry.computeVertexNormals();
          const hasColors = geometry.hasAttribute('color');
          const material = new THREE.MeshStandardMaterial({
            vertexColors: hasColors,
            roughness: 0.85,
            metalness: 0.05,
            color: hasColors ? 0xffffff : 0xcccccc,
            side: THREE.DoubleSide,
            wireframe: config.wireframe,
          });
          const mesh = new THREE.Mesh(geometry, material);
          setupLoadedObject(mesh, 'PLY', sourceFileName);
        },
        onProgress,
        (err) => {
          console.error('PLY load error:', err);
          setErrorMessage('Failed to load PLY model file');
          onLoadingChange(false, 0);
        }
      );
    } else if ((repConfig.type === 'glb' || repConfig.glb_url) && (repConfig.glb_url || (repConfig as any).url)) {
      const glbUrl = (repConfig.glb_url || (repConfig as any).url) as string;
      const sourceFileName = glbUrl.split('/').pop() || 'room_model.glb';
      const gltfLoader = new GLTFLoader();
      gltfLoader.load(
        glbUrl,
        (gltf) => {
          setupLoadedObject(gltf.scene, 'GLB', sourceFileName);
        },
        onProgress,
        (err) => {
          console.error('GLB load error:', err);
          setErrorMessage('Failed to load GLB model file');
          onLoadingChange(false, 0);
        }
      );
    } else {
      setErrorMessage('Selected representation format is not available');
      onLoadingChange(false, 0);
    }
  }, [scene, representation, isIndoor]);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  return (
    <div className="viewport-area" ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Floating Info Overlay (Hidden in Walkthrough mode to avoid clutter) */}
      {!isWalkthrough && (
        <div className="viewport-floating-info">
          <div className="floating-title">{customTitle || scene.display_name}</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
            Representation:{' '}
            <span style={{ color: '#fff', fontWeight: 600 }}>
              {scene.representations[representation]?.label || representation}
            </span>
          </div>
          <GeometryStatsDisplay stats={activeStats} mode="overlay" />
        </div>
      )}

      {/* Walkthrough HUD, Pointer Lock, Telemetry & Controls */}
      {isIndoor && (
        <WalkthroughHUD
          isWalkthroughMode={isWalkthrough}
          onEnterWalkthrough={handleEnterWalkthrough}
          onExitWalkthrough={handleExitWalkthrough}
          playerPos={playerPos}
          playerYaw={playerYaw}
          collisionStatus={collisionStatus}
          fps={fps}
          isPointerLocked={isPointerLocked}
          onRequestPointerLock={() => {
            if (navEngineRef.current) navEngineRef.current.requestPointerLock();
          }}
        />
      )}

      {/* Live Floor Plan Mini-Map */}
      {isIndoor && isWalkthrough && (
        <MiniMapOverlay
          playerPos={playerPos}
          playerYaw={playerYaw}
          isOpen={isMiniMapOpen}
          onToggle={() => setIsMiniMapOpen(!isMiniMapOpen)}
        />
      )}

      {/* Real-Time Source Video Capture Preview */}
      {isIndoor && isWalkthrough && (
        <SourceFramePreview currentFrame={currentFrame} distanceMeters={frameDistance} />
      )}

      {/* Error Banner */}
      {errorMessage && (
        <div className="loading-overlay" style={{ background: 'rgba(15, 23, 42, 0.9)' }}>
          <div style={{ color: 'var(--accent-rose)', fontWeight: 600, fontSize: '0.95rem' }}>{errorMessage}</div>
          {onFallbackToPly && representation === 'textured_obj' && (
            <button className="btn-action primary" onClick={onFallbackToPly} style={{ marginTop: '0.5rem' }}>
              Load PLY Representation Fallback
            </button>
          )}
        </div>
      )}
    </div>
  );
};
