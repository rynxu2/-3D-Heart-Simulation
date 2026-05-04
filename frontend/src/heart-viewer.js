/**
 * Heart Viewer — Photorealistic Cardiac Biomechanics Simulation.
 *
 * Upgraded from v1 (7 mechanics) → v2 (10 mechanics + visual realism):
 *
 * VERTEX SHADER (GPU Cardiac Mechanics):
 *  1. Wiggers-accurate peristaltic contraction wave
 *  2. Differential torsion + rapid diastolic untwist
 *  3. AV-plane descent (asymmetric L/R)
 *  4. Regional wall thickening (non-uniform)
 *  5. Diastolic suction with rapid snap-back
 *  6. Respiratory sway
 *  7. Surface micro-tremor (noise-based)
 *  8. Isovolumetric relaxation phase (NEW)
 *  9. Regional wall motion abnormalities — MI akinesia/dyskinesia (NEW)
 *
 * FRAGMENT SHADER (Visual Realism):
 * 10. Dynamic emissive pulsation (systole/diastole color shift)
 *
 * MATERIAL:
 * - Subsurface scattering via transmission + thickness
 * - Wet sheen layer
 * - Attenuation color for light absorption
 *
 * POST-PROCESSING:
 * - UnrealBloomPass for organic glow
 * - Dynamic bloom strength synced to cardiac cycle
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// Medically accurate cardiac mechanics per condition.
// Includes regional MI parameters for Phase 4.
const CONDITION_CONFIG = {
  normal: {
    squeeze: 1.0,         // Full contraction (EF 55-70%)
    twist: 1.0,           // Normal torsion ~12°
    tremor: 0.0,          // No tremor — healthy rhythm
    color: 0x00d68f,      // Green — healthy
    infarctSeverity: 0.0, // No infarction
    infarctAngle: 0.0,    // N/A
    infarctWidth: 0.5,    // N/A
  },
  abnormal: {
    squeeze: 0.80,        // Reduced EF (35-55%) — weaker pump
    twist: 1.3,           // Compensatory increased torsion
    tremor: 0.15,         // Mild tremor from ectopic beats
    color: 0xf59e0b,      // Yellow — warning
    infarctSeverity: 0.0,
    infarctAngle: 0.0,
    infarctWidth: 0.5,
  },
  infarction: {
    squeeze: 0.65,        // Globally reduced EF (25-40%)
    twist: 1.5,           // Compensatory torsion
    tremor: 0.30,         // Fibrillation-like tremor
    color: 0xff2020,      // Red — critical
    infarctSeverity: 0.85, // Severe regional damage
    infarctAngle: -0.8,    // Anterior wall (LAD territory, ~-45°)
    infarctWidth: 0.6,     // Moderate zone width
  },
  pain: {
    squeeze: 0.65,
    twist: 1.5,
    tremor: 0.30,
    color: 0xff4757,
    infarctSeverity: 0.85,
    infarctAngle: -0.8,
    infarctWidth: 0.6,
  },
  no_pain: {
    squeeze: 1.0,
    twist: 1.0,
    tremor: 0.0,
    color: 0x00d68f,
    infarctSeverity: 0.0,
    infarctAngle: 0.0,
    infarctWidth: 0.5,
  },
};

// ═══════════════════════════════════════════════════════════
// GLSL VERTEX SHADER — Wiggers-accurate cardiac cycle
// Phase 1: Timing recalibrated to medical Wiggers diagram
// Phase 4: Regional wall motion abnormalities (MI akinesia)
// ═══════════════════════════════════════════════════════════
const CARDIAC_VERTEX_GLSL = /* glsl */ `
#include <begin_vertex>

// ─── Cardiac cycle phase (0.0 → 1.0) ───
float t = fract(uTime * uBPS);

// ─── Wiggers-accurate Gaussian pulses ───
// Calibrated to 75 BPM (800ms cycle):
//   Atrial systole:        0–80ms   → t = 0.00–0.10
//   Isovolumetric contract: 80–130ms → t = 0.10–0.16
//   Ventricular ejection:  130–360ms → t = 0.16–0.45
//   Isovolumetric relax:   360–415ms → t = 0.45–0.52 (NEW)
//   Rapid filling:         415–520ms → t = 0.52–0.65
//   Slow filling:          520–720ms → t = 0.65–0.90
//   Atrial kick:           720–800ms → t = 0.90–1.00

float atrialSystole    = exp(-pow((t - 0.05) / 0.030, 2.0));
float isoContraction   = exp(-pow((t - 0.13) / 0.018, 2.0));
float ventricularEject = exp(-pow((t - 0.30) / 0.070, 2.0));
float isoRelaxation    = exp(-pow((t - 0.48) / 0.020, 2.0));
float rapidFilling     = exp(-pow((t - 0.58) / 0.035, 2.0));
float slowFilling      = exp(-pow((t - 0.75) / 0.080, 2.0));
float atrialKick       = exp(-pow((t - 0.95) / 0.025, 2.0));

float systole  = isoContraction + ventricularEject;
float diastole = rapidFilling + slowFilling;

// ─── Normalize Y by actual model height ───
float heightNorm = clamp(transformed.y * uInvHeight, -1.0, 1.0);
float atrialZone    = smoothstep(0.2, 0.6, heightNorm);
float ventricleZone = smoothstep(0.3, -0.3, heightNorm);
float apexZone      = smoothstep(-0.2, -0.8, heightNorm);

// ─── Phase 4: REGIONAL WALL MOTION ABNORMALITY ───
// Compute angular position of this vertex (for MI zone masking)
float vertexAngle = atan(transformed.z, transformed.x);
float infarctMask = exp(-pow((vertexAngle - uInfarctAngle) / uInfarctWidth, 2.0));
// infarctFactor: 1.0 = healthy, 0.0 = fully akinetic, <0.0 = dyskinetic
float infarctFactor = 1.0 - infarctMask * uInfarctSeverity;
// Dyskinesia: damaged zone bulges outward during systole
float dyskinesia = infarctMask * uInfarctSeverity * ventricularEject * 0.04;

// ─── 1. PERISTALTIC WAVE (Wiggers-calibrated) ───
float localSqueeze = atrialSystole * atrialZone * 0.018
                   + isoContraction * ventricleZone * 0.035
                   + ventricularEject * ventricleZone * 0.090
                   + atrialKick * atrialZone * 0.008;

// Apply regional MI: healthy zones contract, damaged zones don't
localSqueeze *= max(infarctFactor, 0.0);

float sqXZ = 1.0 - localSqueeze * uSqueezeIntensity
           + isoRelaxation * 0.015
           + rapidFilling * 0.012;

float sqY = 1.0
  - atrialSystole * atrialZone * 0.025 * uSqueezeIntensity * max(infarctFactor, 0.0)
  - ventricularEject * ventricleZone * 0.070 * uSqueezeIntensity * max(infarctFactor, 0.0)
  + rapidFilling * 0.010
  + isoRelaxation * 0.005;

transformed.x *= sqXZ;
transformed.z *= sqXZ;
transformed.y *= sqY;

// Dyskinesia: bulge outward at damaged region
transformed += normal * dyskinesia;

// ─── 2. DIFFERENTIAL TORSION + DIASTOLIC UNTWIST ───
float twistDir = -heightNorm;
// Systolic twist (wringing motion)
float twistSystole = (ventricularEject * 0.10 + isoContraction * 0.04) * uTwistIntensity;
// Diastolic rapid untwist (energy release during early diastole)
float untwist = rapidFilling * 0.06 * uTwistIntensity;
float twistAngle = (twistSystole - untwist) * twistDir;

// Apply regional MI: damaged zone has reduced torsion
twistAngle *= mix(1.0, 0.3, infarctMask * uInfarctSeverity);

float cs = cos(twistAngle);
float sn = sin(twistAngle);
float rx = transformed.x * cs - transformed.z * sn;
float rz = transformed.x * sn + transformed.z * cs;
transformed.x = rx;
transformed.z = rz;

// ─── 3. AV-PLANE DESCENT (asymmetric L/R) ───
float avZone = 1.0 - abs(heightNorm);
float descent = ventricularEject * avZone * 0.08 * uSqueezeIntensity / uInvHeight;
// Asymmetry: left ventricle descends ~15% more than right
float lrBias = 1.0 + 0.15 * step(0.0, transformed.x);
transformed.y -= descent * uInvHeight * lrBias;

// ─── 4. WALL THICKENING (non-uniform — apex thickens more) ───
float apexBias = 1.0 + apexZone * 0.4;
float thickening = ventricularEject * ventricleZone * 0.028 * uSqueezeIntensity / uInvHeight;
thickening *= apexBias * max(infarctFactor, 0.0);
transformed += normal * thickening * uInvHeight;

// ─── 5. DIASTOLIC SUCTION (rapid snap-back) ───
float suction = rapidFilling * ventricleZone * 0.018 / uInvHeight;
// Snap-back is stronger than v1 — more realistic elastic recoil
float snapBack = isoRelaxation * ventricleZone * 0.008 / uInvHeight;
transformed.x += normal.x * (suction + snapBack) * uInvHeight;
transformed.z += normal.z * (suction + snapBack) * uInvHeight;

// ─── 6. RESPIRATORY SWAY ───
float breathScale = 1.0 / uInvHeight * 0.002;
transformed.x += sin(uTime * 1.57) * breathScale;
transformed.y += sin(uTime * 1.57 + 1.5) * breathScale * 0.5;

// ─── 7. SURFACE MICRO-TREMOR (improved noise-like pattern) ───
float freqScale = uInvHeight * 3.0;
// Multi-frequency noise approximation instead of simple sine
float tremor = (
    sin(transformed.x * freqScale + uTime * 12.0)
  * sin(transformed.y * freqScale * 1.3 + uTime * 9.0)
  + sin(transformed.z * freqScale * 0.7 + uTime * 15.0) * 0.5
) * 0.002 * uTremor * systole / uInvHeight;
// Tremor is stronger in the infarct zone (fibrillation)
float tremBias = 1.0 + infarctMask * uInfarctSeverity * 2.0;
transformed += normal * tremor * uInvHeight * tremBias;
`;

// Fragment shader emissive pulsation removed — static lighting only

export class HeartViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.heartMesh = null;
    this.bpm = 72;
    this.condition = 'normal';
    this.clock = new THREE.Clock();
    this.composer = null;
    this.bloomPass = null;

    this.shaderUniforms = {
      uTime: { value: 0 },
      uBPS: { value: 72 / 60 },
      uSqueezeIntensity: { value: 1.0 },
      uTwistIntensity: { value: 1.0 },
      uTremor: { value: 0.0 },
      uInvHeight: { value: 0.5 },
      // Phase 4: Regional MI
      uInfarctSeverity: { value: 0.0 },
      uInfarctAngle: { value: 0.0 },
      uInfarctWidth: { value: 0.5 },
    };

    this._init();
    this._loadModel();
  }

  _init() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight || 400;

    this.scene = new THREE.Scene();

    // Depth background — radial gradient sphere (dark center → darker edges)
    this._setupDepthBackground();

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    this.camera.position.set(0, 2, 15);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // ── Lighting — neutral whites only, no colored tints ──
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
    keyLight.position.set(8, 10, 8);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    fillLight.position.set(-5, -3, 4);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
    rimLight.position.set(-2, 5, -8);
    this.scene.add(rimLight);

    // heartGroup receives momentum animation (position kicks)
    this.heartGroup = new THREE.Group();
    this.scene.add(this.heartGroup);

    // ── Post-Processing (Bloom) ──
    this._setupBloom(w, h);

    new ResizeObserver(() => this._onResize()).observe(this.container);
    this._animate();
  }

  _setupDepthBackground() {
    // 1. Pure black base
    this.scene.background = new THREE.Color(0x020202);

    // 2. Subtle depth glow from behind
    const centerDepthGlow = new THREE.PointLight(0x39ff14, 0.8, 30);
    centerDepthGlow.position.set(0, 0, -10);
    this.scene.add(centerDepthGlow);

    // (Scanner rings removed as requested)

    // 4. Floating Holographic Particles (Provides massive parallax depth)
    const pGeo = new THREE.BufferGeometry();
    const pCount = 300;
    const pPos = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount * 3; i++) {
      pPos[i] = (Math.random() - 0.5) * 40; // Spread in 40x40x40 volume
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    
    const pMat = new THREE.PointsMaterial({
      size: 0.08,
      color: 0x39ff14,
      transparent: true,
      opacity: 0.3,
      sizeAttenuation: true
    });
    
    this.bgParticles = new THREE.Points(pGeo, pMat);
    this.scene.add(this.bgParticles);
  }

  _setupBloom(w, h) {
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(w, h),
      0.5,    // strength — subtle organic glow
      0.8,    // radius — wide soft spread
      0.72    // threshold — only brightest areas bloom
    );
    this.composer.addPass(this.bloomPass);
  }

  _loadModel() {
    const loader = new GLTFLoader();
    loader.load(
      '/models/heart.glb',
      (gltf) => {
        const model = gltf.scene;

        // 1. Center model
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        model.position.sub(center);

        // 2. Wrapper group
        const wrapperGroup = new THREE.Group();
        wrapperGroup.add(model);

        // 3. Normalize scale + compute shader height factor
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 9 / maxDim;
        wrapperGroup.scale.set(scale, scale, scale);

        const halfHeight = size.y / 2.0;
        this.shaderUniforms.uInvHeight.value = 1.0 / Math.max(halfHeight, 0.01);

        // 4. Material — preserve original texture, no color overlays
        model.traverse((child) => {
          if (!child.isMesh) return;

          const oldMat = child.material;
          child.material = new THREE.MeshPhysicalMaterial({
            map: oldMat.map,
            color: oldMat.color || new THREE.Color(0xffffff),
            vertexColors: oldMat.vertexColors,
            side: THREE.DoubleSide,

            // PBR base — neutral, no color tinting
            roughness: 0.4,
            metalness: 0.0,

            // Clearcoat for wet surface look
            clearcoat: 0.3,
            clearcoatRoughness: 0.2,
          });

          // ── Inject GLSL into both vertex + fragment shaders ──
          child.material.onBeforeCompile = (shader) => {
            // Register all uniforms
            for (const [key, val] of Object.entries(this.shaderUniforms)) {
              shader.uniforms[key] = val;
            }

            // ── Uniform declarations (shared by vertex + fragment) ──
            const uniformBlock = `
              uniform float uTime;
              uniform float uBPS;
              uniform float uSqueezeIntensity;
              uniform float uTwistIntensity;
              uniform float uTremor;
              uniform float uInvHeight;
              uniform float uInfarctSeverity;
              uniform float uInfarctAngle;
              uniform float uInfarctWidth;
            `;

            // ── VERTEX SHADER: Cardiac mechanics ──
            shader.vertexShader = uniformBlock + shader.vertexShader;
            shader.vertexShader = shader.vertexShader.replace(
              '#include <begin_vertex>',
              CARDIAC_VERTEX_GLSL
            );

          };
        });

        // 5. Add to heartGroup for momentum animation
        this.heartGroup.add(wrapperGroup);
        this.heartMesh = model;
        console.log('✅ Heart v2 loaded — 9 cardiac mechanics + SSS + bloom');
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

    // Cardiac mechanics
    this.shaderUniforms.uSqueezeIntensity.value = cfg.squeeze;
    this.shaderUniforms.uTwistIntensity.value = cfg.twist;
    this.shaderUniforms.uTremor.value = cfg.tremor;

    // Phase 4: Regional MI
    this.shaderUniforms.uInfarctSeverity.value = cfg.infarctSeverity;
    this.shaderUniforms.uInfarctAngle.value = cfg.infarctAngle;
    this.shaderUniforms.uInfarctWidth.value = cfg.infarctWidth;

  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    const elapsed = this.clock.getElapsedTime();
    this.shaderUniforms.uTime.value = elapsed;

    if (this.heartGroup) {
      const bps = this.bpm / 60;
      const t = (elapsed * bps) % 1.0;

      // Wiggers-synced ejection pulse
      const vContraction = Math.exp(-Math.pow((t - 0.30) / 0.07, 2));
      // Isovolumetric relaxation rebound
      const isoRelax = Math.exp(-Math.pow((t - 0.48) / 0.02, 2));

      // Momentum — whole group kicks down and back during ejection
      this.heartGroup.position.y = -(vContraction * 0.25) + (isoRelax * 0.08);
      this.heartGroup.position.z = -(vContraction * 0.10);
    }

    // Slowly rotate background elements for dynamic sci-fi feel
    if (this.bgParticles) {
      this.bgParticles.rotation.y = elapsed * 0.02;
      this.bgParticles.rotation.x = elapsed * 0.01;
    }

    // Render via post-processing composer (includes bloom)
    if (this.composer) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    if (this.composer) {
      this.composer.setSize(w, h);
    }
  }
}
