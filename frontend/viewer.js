import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export function createViewer(container) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  } catch {
    throw new Error("WebGL 2 başlatılamadı. Modeli indirme düğmeleriyle kaydedebilirsiniz.");
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.14;
  container.append(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", "Sürükleyerek döndürülebilen 3D model");

  const scene = new THREE.Scene();
  const environment = new RoomEnvironment();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(environment, 0.04).texture;
  scene.environmentIntensity = 0.62;
  environment.dispose();
  pmrem.dispose();
  const key = new THREE.DirectionalLight(0xffeee1, 3.1);
  scene.add(key, new THREE.HemisphereLight(0xe1dfff, 0x565a75, 1.0));
  const camera = new THREE.PerspectiveCamera(37, 1, 0.02, 60);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = true;
  controls.enableDamping = false;
  controls.minDistance = 0.3;
  controls.maxDistance = 14;
  const loader = new GLTFLoader();
  let mesh;
  let currentVersion = 0;
  let wireframeEnabled = false;

  function render() {
    renderer.render(scene, camera);
  }

  function reset() {
    camera.position.set(1.8, 1.12, 2.4);
    controls.target.set(0, 0, 0);
    controls.update();
    render();
  }

  function view(name) {
    if (!mesh) return;
    const directions = { front: [0, 0, 3.2], right: [-3.2, 0, 0], back: [0, 0, -3.2], left: [3.2, 0, 0] };
    camera.position.set(...directions[name]);
    controls.target.set(0, 0, 0);
    controls.update();
    render();
  }

  function light(degrees) {
    const angle = THREE.MathUtils.degToRad(degrees);
    key.position.set(Math.cos(angle) * 3, 2.6, Math.sin(angle) * 3);
    scene.environmentRotation.y = angle;
    render();
  }

  function wireframe(active) {
    wireframeEnabled = active;
    mesh?.traverse((object) => {
      if (!object.isMesh) return;
      for (const material of [].concat(object.material)) material.wireframe = active;
    });
    render();
  }

  function dispose(root) {
    const textures = new Set();
    root.traverse((object) => {
      if (!object.isMesh) return;
      object.geometry.dispose();
      for (const material of [].concat(object.material)) {
        for (const property of Object.values(material)) {
          if (property?.isTexture) textures.add(property);
        }
        material.dispose();
      }
    });
    for (const texture of textures) {
      texture.image?.close?.();
      texture.dispose();
    }
  }

  function clear() {
    currentVersion++;
    if (mesh) {
      scene.remove(mesh);
      dispose(mesh);
      mesh = undefined;
      render();
    }
  }

  async function open(url) {
    clear();
    const version = currentVersion;
    let gltf;
    try {
      gltf = await loader.loadAsync(url);
    } catch {
      throw new Error("GLB önizlemesi yüklenemedi. Modeli GLB düğmesiyle kaydedebilirsiniz.");
    }
    if (currentVersion !== version) {
      dispose(gltf.scene);
      return false;
    }
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    if (sphere.isEmpty() || sphere.radius <= 0) {
      dispose(gltf.scene);
      throw new Error("GLB dosyasında görüntülenebilir yüzey bulunamadı.");
    }
    mesh = new THREE.Group();
    const scale = 0.95 / sphere.radius;
    mesh.add(gltf.scene);
    mesh.scale.setScalar(scale);
    mesh.position.copy(sphere.center).multiplyScalar(-scale);
    scene.add(mesh);
    wireframe(wireframeEnabled);
    reset();
    return true;
  }

  controls.addEventListener("change", render);
  new ResizeObserver(() => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    render();
  }).observe(container);
  light(40);
  reset();
  return { open, reset, clear, wireframe, light, view };
}
