/**
 * Heart Viewer — Advanced Cardiac Biomechanics via GPU Vertex Shader.
 *
 * Simulates 7 real cardiac mechanics on the GPU:
 * 1. Peristaltic contraction wave (atria → ventricles, top → bottom)
 * 2. Differential torsion (apex twists opposite to base)
 * 3. AV-plane descent (mitral annulus drops during systole)
 * 4. Wall thickening (myocardium pushes outward along normals)
 * 5. Diastolic suction (rapid snap-back in early diastole)
 * 6. Respiratory sway (subtle side-to-side from breathing)
 * 7. Surface micro-tremor (fibrillation-like fine detail)
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const CONDITION_CONFIG = {
  normal:     { squeeze: 1.0, twist: 1.0, tremor: 0.0, color: 0x00d68f },
  abnormal:   { squeeze: 1.1, twist: 1.2, tremor: 0.5, color: 0xf59e0b },
  infarction: { squeeze: 1.4, twist: 1.5, tremor: 0.3, color: 0xff2020 },
  pain:       { squeeze: 1.4, twist: 1.5, tremor: 0.3, color: 0xff4757 },  // alias
  no_pain:    { squeeze: 1.0, twist: 1.0, tremor: 0.0, color: 0x00d68f },  // alias
};

// ═══════════════════════════════════════════════════════════
// GLSL VERTEX SHADER — injected into Three.js material
// ═══════════════════════════════════════════════════════════
const CARDIAC_GLSL = /* glsl */ `
#include <begin_vertex>

// ─── Cardiac cycle phase (0.0 → 1.0) ───
float t = fract(uTime * uBPS);

// ─── Gaussian pulse per cardiac phase ───
float atrialSystole      = exp(-pow((t - 0.05) / 0.025, 2.0));
float isoContraction     = exp(-pow((t - 0.12) / 0.012, 2.0));
float ventricularEject   = exp(-pow((t - 0.25) / 0.050, 2.0));
float valveClose         = exp(-pow((t - 0.38) / 0.015, 2.0));
float rapidFilling       = exp(-pow((t - 0.48) / 0.030, 2.0));
float slowFilling        = exp(-pow((t - 0.70) / 0.080, 2.0));

float systole = isoContraction + ventricularEject;

// ─── Normalize Y by actual model height ───
float heightNorm = clamp(transformed.y * uInvHeight, -1.0, 1.0);
float atrialZone = smoothstep(0.2, 0.6, heightNorm);
float ventricleZone = smoothstep(0.3, -0.3, heightNorm);

// ─── 1. PERISTALTIC WAVE (scale-independent — uses ratios) ───
float localSqueeze = atrialSystole * atrialZone * 0.015
                   + isoContraction * ventricleZone * 0.030
                   + ventricularEject * ventricleZone * 0.085;

float sqXZ = 1.0 - localSqueeze * uSqueezeIntensity
           + valveClose * 0.012
           + rapidFilling * 0.010;

float sqY = 1.0
  - atrialSystole * atrialZone * 0.025 * uSqueezeIntensity
  - ventricularEject * ventricleZone * 0.065 * uSqueezeIntensity
  + rapidFilling * 0.008;

transformed.x *= sqXZ;
transformed.z *= sqXZ;
transformed.y *= sqY;

// ─── 2. DIFFERENTIAL TORSION (angle-based — scale-independent) ───
float twistDir = -heightNorm;
float twistMag = (ventricularEject * 0.10 + isoContraction * 0.04) * uTwistIntensity;
float twistAngle = twistMag * twistDir;

float cs = cos(twistAngle);
float sn = sin(twistAngle);
float rx = transformed.x * cs - transformed.z * sn;
float rz = transformed.x * sn + transformed.z * cs;
transformed.x = rx;
transformed.z = rz;

// ─── 3. AV-PLANE DESCENT (scaled by model height) ───
float avZone = 1.0 - abs(heightNorm);
float descent = ventricularEject * avZone * 0.08 * uSqueezeIntensity / uInvHeight;
transformed.y -= descent * uInvHeight;

// ─── 4. WALL THICKENING (scaled by model size) ───
float thickening = ventricularEject * ventricleZone * 0.025 * uSqueezeIntensity / uInvHeight;
transformed += normal * thickening * uInvHeight;

// ─── 5. DIASTOLIC SUCTION (scaled by model size) ───
float suction = rapidFilling * ventricleZone * 0.015 / uInvHeight;
transformed.x += normal.x * suction * uInvHeight;
transformed.z += normal.z * suction * uInvHeight;

// ─── 6. RESPIRATORY SWAY (scaled by model size) ───
float breathScale = 1.0 / uInvHeight * 0.002;
transformed.x += sin(uTime * 1.57) * breathScale;
transformed.y += sin(uTime * 1.57 + 1.5) * breathScale * 0.5;

// ─── 7. SURFACE MICRO-TREMOR (scaled by model size) ───
float freqScale = uInvHeight * 3.0;
float tremor = sin(transformed.x * freqScale + uTime * 12.0)
             * sin(transformed.y * freqScale + uTime * 9.0)
             * 0.003 * uTremor * systole / uInvHeight;
transformed += normal * tremor * uInvHeight;
`;

export class HeartViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.heartMesh = null;
    this.bpm = 72;
    this.condition = 'normal';
    this.clock = new THREE.Clock();
    this.baseY = 0;
    this.baseZ = 0;

    this.shaderUniforms = {
      uTime: { value: 0 },
      uBPS: { value: 72 / 60 },
      uSqueezeIntensity: { value: 1.0 },
      uTwistIntensity: { value: 1.0 },
      uTremor: { value: 0.0 },
      uInvHeight: { value: 0.5 }, // 1.0 / halfHeight — set after model loads
    };

    this._init();
    this._loadModel();
  }

  _init() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight || 400;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    this.camera.position.set(0, 2, 15);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    const mainLight = new THREE.DirectionalLight(0xffffff, 2);
    mainLight.position.set(10, 10, 10);
    this.scene.add(mainLight);

    const blueRim = new THREE.PointLight(0x3366ff, 5, 20);
    blueRim.position.set(-10, 5, -5);
    this.scene.add(blueRim);

    this.glowLight = new THREE.PointLight(0x00d68f, 0.4, 10);
    this.glowLight.position.set(0, 0, 2);
    this.scene.add(this.glowLight);

    // heartGroup receives momentum animation (position kicks)
    this.heartGroup = new THREE.Group();
    this.scene.add(this.heartGroup);

    new ResizeObserver(() => this._onResize()).observe(this.container);
    this._animate();
  }

  _loadModel() {
    const loader = new GLTFLoader();
    loader.load(
      '/models/anatomical_heart_right_dominance.glb',
      (gltf) => {
        const model = gltf.scene;

        // 1. Center model
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        model.position.sub(center);

        // 2. Wrapper group — fixes coordinate scaling bug
        const wrapperGroup = new THREE.Group();
        wrapperGroup.add(model);

        // 3. Normalize scale + compute shader height factor
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 7 / maxDim;
        wrapperGroup.scale.set(scale, scale, scale);

        // Pass inverse half-height so shader can normalize Y regardless of model size
        const halfHeight = size.y / 2.0;
        this.shaderUniforms.uInvHeight.value = 1.0 / Math.max(halfHeight, 0.01);

        // 4. Material upgrade + shader injection
        model.traverse((child) => {
          if (!child.isMesh) return;

          const oldMat = child.material;
          child.material = new THREE.MeshPhysicalMaterial({
            map: oldMat.map,
            color: oldMat.color || new THREE.Color(0xcc0000),
            vertexColors: oldMat.vertexColors,
            side: THREE.DoubleSide,
            roughness: 0.15,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1,
          });

          // Inject cardiac GLSL
          child.material.onBeforeCompile = (shader) => {
            for (const [key, val] of Object.entries(this.shaderUniforms)) {
              shader.uniforms[key] = val;
            }

            shader.vertexShader = `
              uniform float uTime;
              uniform float uBPS;
              uniform float uSqueezeIntensity;
              uniform float uTwistIntensity;
              uniform float uTremor;
              uniform float uInvHeight;
              ${shader.vertexShader}
            `;

            shader.vertexShader = shader.vertexShader.replace(
              '#include <begin_vertex>',
              CARDIAC_GLSL
            );
          };
        });

        // 5. Add to heartGroup for momentum animation
        this.heartGroup.add(wrapperGroup);
        this.heartMesh = model;
        console.log('✅ Heart loaded — 7 cardiac mechanics active on GPU');
      },
      (xhr) => {
        if (xhr.total > 0) {
          console.log(`${Math.round(xhr.loaded / xhr.total * 100)}% loaded`);
        }
      },
      (err) => console.error('❌ Model error:', err)
    );
  }

  setBPM(bpm) {
    this.bpm = Math.max(30, Math.min(180, bpm));
    this.shaderUniforms.uBPS.value = this.bpm / 60;
  }

  setCondition(condition) {
    this.condition = condition;
    const cfg = CONDITION_CONFIG[condition] || CONDITION_CONFIG.normal;
    this.shaderUniforms.uSqueezeIntensity.value = cfg.squeeze;
    this.shaderUniforms.uTwistIntensity.value = cfg.twist;
    this.shaderUniforms.uTremor.value = cfg.tremor;
    this.glowLight.color.setHex(cfg.color);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    const elapsed = this.clock.getElapsedTime();
    this.shaderUniforms.uTime.value = elapsed;

    if (this.heartGroup) {
      const bps = this.bpm / 60;
      const t = (elapsed * bps) % 1.0;
      const vContraction = Math.exp(-Math.pow((t - 0.25) / 0.05, 2));

      // Momentum — whole group kicks down and back during ejection
      this.heartGroup.position.y = -(vContraction * 0.25);
      this.heartGroup.position.z = -(vContraction * 0.1);

      // Glow tracks systole
      this.glowLight.intensity = 0.2 + vContraction * 0.7;
    }

    this.renderer.render(this.scene, this.camera);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
