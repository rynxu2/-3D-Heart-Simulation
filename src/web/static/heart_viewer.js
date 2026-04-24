class HeartViewer {
    constructor(id, cfg = {}) {
        this.container = document.getElementById(id);
        if (!this.container) return;
        this.config = {
            bg: cfg.backgroundColor || '#1a1a2e',
            ambient: cfg.ambientLight || 0.4,
            dirLight: cfg.directionalLight || 0.8,
            camDist: cfg.cameraDistance || 5,
            ...cfg
        };
        this.heartbeat = null;
        this.heartMesh = null;
        this.damageMarkers = [];
        this.init()
    }
    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.bg);
        const a = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, a, 0.1, 100);
        this.camera.position.set(0, 0, this.config.camDist);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        this.scene.add(new THREE.AmbientLight(0xffffff, this.config.ambient));
        const d = new THREE.DirectionalLight(0xffffff, this.config.dirLight);
        d.position.set(5, 5, 5);
        this.scene.add(d);
        this.scene.add(new THREE.DirectionalLight(0x4488ff, 0.3).position.set(-3, 2, -3) || new THREE.DirectionalLight(0x4488ff, 0.3));
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.clock = new THREE.Clock();
        window.addEventListener('resize', () => this.onResize());
        this.animate()
    }
    loadModel(path) {
        const loader = new THREE.GLTFLoader();
        return new Promise((res, rej) => {
            loader.load(path, g => {
                if (this.heartMesh) this.scene.remove(this.heartMesh);
                this.heartMesh = g.scene;
                const b = new THREE.Box3().setFromObject(this.heartMesh);
                const c = b.getCenter(new THREE.Vector3());
                this.heartMesh.position.sub(c);
                const s = b.getSize(new THREE.Vector3());
                this.heartMesh.scale.setScalar(2 / Math.max(s.x, s.y, s.z));
                this.scene.add(this.heartMesh);
                res(this.heartMesh)
            }, undefined, rej)
        })
    }
    createSimpleHeart(color = '#e74c3c') {
        if (this.heartMesh) this.scene.remove(this.heartMesh);
        const g = new THREE.Group();
        const m = new THREE.MeshPhongMaterial({
            color: new THREE.Color(color),
            shininess: 60,
            transparent: true,
            opacity: 0.9
        });
        const body = new THREE.Mesh(new THREE.SphereGeometry(0.8, 32, 32), m);
        body.scale.set(1, 1.1, 0.9);
        g.add(body);
        const lb = new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 32), m.clone());
        lb.position.set(-0.35, 0.55, 0);
        g.add(lb);
        const rb = new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 32), m.clone());
        rb.position.set(0.35, 0.55, 0);
        g.add(rb);
        const cn = new THREE.Mesh(new THREE.ConeGeometry(0.6, 0.8, 32), m.clone());
        cn.position.set(0, -0.9, 0);
        g.add(cn);
        this.heartMesh = g;
        this.scene.add(g)
    }
    setHeartbeatParams(p) {
        this.heartbeat = {
            bpm: p.bpm || 72,
            cycleDuration: p.cycleDuration || (60 / 72),
            systoleRatio: p.systoleRatio || 0.33,
            contractionMin: p.contractionMin || 0.85,
            contractionMax: p.contractionMax || 1.0,
            jitter: p.jitter || 0,
            pattern: p.pattern || 'regular'
        }
    }
    addDamageZone(pos, r, color = '#2c2c2c') {
        const mk = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 16), new THREE.MeshPhongMaterial({
            color: new THREE.Color(color),
            transparent: true,
            opacity: 0.7,
            emissive: new THREE.Color('#330000'),
            emissiveIntensity: 0.5
        }));
        mk.position.set(...pos);
        if (this.heartMesh) this.heartMesh.add(mk);
        this.damageMarkers.push(mk)
    }
    animate() {
        requestAnimationFrame(() => this.animate());
        const t = this.clock.getElapsedTime(); 
        if (this.heartMesh && this.heartbeat) { 
            const s = this.calcScale(t); 
            this.heartMesh.scale.setScalar(s); 
            const i = 1 - (s - this.heartbeat.contractionMin) / (this.heartbeat.contractionMax - this.heartbeat.contractionMin); 
            this.heartMesh.traverse(c => { 
                if (c.isMesh && c.material && c.material.emissiveIntensity !== undefined) c.material.emissiveIntensity = i * 0.3 }); 
                this.damageMarkers.forEach(m => { m.material.emissiveIntensity = 0.3 + i * 0.5 }) 
            } 
            this.controls.update(); 
            this.renderer.render(this.scene, this.camera)
    }
    calcScale(t) { 
        if (!this.heartbeat) return 1; 
        let cy = this.heartbeat.cycleDuration; 
        if (this.heartbeat.jitter > 0) { 
            const sd = Math.floor(t * 10); 
            const j = (Math.sin(sd * 12.9898 + sd * 78.233) * 43758.5453) % 1; 
            cy *= (1 + (j - 0.5) * 2 * this.heartbeat.jitter) } 
            const p = (t % cy) / cy; 
            const sr = this.heartbeat.systoleRatio; 
            const mn = this.heartbeat.contractionMin; 
            const mx = this.heartbeat.contractionMax; 
            if (p < sr) { 
                const pr = p / sr; 
                const e = 0.5 - 0.5 * Math.cos(pr * Math.PI); 
                return mx - e * (mx - mn) 
            } else { const pr = (p - sr) / (1 - sr); const e = 0.5 - 0.5 * Math.cos(pr * Math.PI); return mn + e * (mx - mn) } 
    }
    updateColor(hex) { 
        if (!this.heartMesh) return; 
        const c = new THREE.Color(hex); 
        this.heartMesh.traverse(ch => { 
            if (ch.isMesh && ch.material) ch.material.color = c 
        })
    }
    onResize() { 
        const w = this.container.clientWidth, h = this.container.clientHeight; 
        this.camera.aspect = w / h; 
        this.camera.updateProjectionMatrix(); 
        this.renderer.setSize(w, h) 
    }
    dispose() { 
        this.renderer.dispose(); 
        this.controls.dispose() 
    }
}
if (typeof module !== 'undefined') module.exports = HeartViewer;
