import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// --- 1. KHỞI TẠO SCENE & RENDERER ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2, 15); // Đặt camera đủ xa để nhìn thấy quả tim sau khi scale

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// --- 2. ÁNH SÁNG (Dùng để làm nổi bật bề mặt ướt) ---
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

const mainLight = new THREE.DirectionalLight(0xffffff, 2);
mainLight.position.set(10, 10, 10);
mainLight.castShadow = true;
scene.add(mainLight);

const blueRim = new THREE.PointLight(0x3366ff, 5, 20); // Đèn xanh tạo điểm nhấn viền
blueRim.position.set(-10, 5, -5);
scene.add(blueRim);

// --- 3. BIẾN ĐIỀU KHIỂN SHADER ---
const shaderParams = { uTime: { value: 0 } };

// --- 4. TẢI VÀ XỬ LÝ MÔ HÌNH ---
let heartGroup = new THREE.Group(); 
scene.add(heartGroup);

const loader = new GLTFLoader();
loader.load(
    './anatomical_heart_-_codominance.glb', // Đảm bảo file nằm trong thư mục public (nếu dùng Vite)
    (gltf) => {
        const model = gltf.scene;

        // 1. Tính toán tâm và kích thước gốc
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        // 2. Dời tâm quả tim về chính giữa
        model.position.sub(center);

        // 3. FIX LỖI: Đưa vào một Group trung gian để không bị văng tọa độ khi Scale
        const wrapperGroup = new THREE.Group();
        wrapperGroup.add(model);

        // 4. Thu nhỏ Group trung gian đó
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 7 / maxDim; 
        wrapperGroup.scale.set(scale, scale, scale);

        // 5. Cài đặt Vật liệu & Shader
        model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;

                const oldMat = child.material;
                child.material = new THREE.MeshPhysicalMaterial({
                    map: oldMat.map,
                    color: oldMat.color || new THREE.Color(0xcc0000), 
                    vertexColors: oldMat.vertexColors, // RẤT QUAN TRỌNG: Sửa lỗi model bị đen thui
                    side: THREE.DoubleSide,            // Hiển thị cả mặt trong và ngoài
                    roughness: 0.15,
                    clearcoat: 1.0,
                    clearcoatRoughness: 0.1
                });

                // Tiêm Shader nhịp đập
                child.material.onBeforeCompile = (shader) => {
                    shader.uniforms.uTime = shaderParams.uTime;
                    shader.vertexShader = `uniform float uTime;\n${shader.vertexShader}`;
                    shader.vertexShader = shader.vertexShader.replace(
                        '#include <begin_vertex>',
                        `
                        #include <begin_vertex>
                        float bps = 75.0 / 60.0;
                        float t = fract(uTime * bps);
                        float vContraction = exp(-pow((t - 0.25) / 0.05, 2.0));
                        
                        float sXZ = 1.0 - (vContraction * 0.1);
                        float sY = 1.0 - (vContraction * 0.08);
                        transformed.x *= sXZ;
                        transformed.z *= sXZ;
                        transformed.y *= sY;

                        float angle = vContraction * 0.15;
                        float c = cos(angle);
                        float s = sin(angle);
                        mat2 rot = mat2(c, s, -s, c);
                        transformed.xz = rot * transformed.xz;
                        `
                    );
                };
            }
        });

        // Thêm hộp chứa model vào heartGroup để chạy animation quán tính
        heartGroup.add(wrapperGroup);
        console.log("Đã tải và fix lỗi văng tọa độ thành công!");
    },
    (xhr) => console.log( Math.round(xhr.loaded / xhr.total * 100) + '% loaded'),
    (err) => console.error("Lỗi tải file:", err)
);

// --- 5. ANIMATION LOOP ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getElapsedTime();

    // Cập nhật thời gian cho Shader
    shaderParams.uTime.value = delta;

    // Hiệu ứng giật quán tính (Momentum)
    if (heartGroup) {
        const t = (delta * (75 / 60)) % 1.0;
        const vContraction = Math.exp(-Math.pow((t - 0.25) / 0.05, 2));

        // Tim giật nhẹ xuống dưới và ra sau khi bóp
        heartGroup.position.y = -(vContraction * 0.25);
        heartGroup.position.z = -(vContraction * 0.1);
    }

    controls.update();
    renderer.render(scene, camera);
}

animate();

// Xử lý resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});