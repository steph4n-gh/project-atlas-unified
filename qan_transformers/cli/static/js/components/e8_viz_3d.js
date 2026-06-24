/**
 * GossetGate: E8 Concentric Shells 3D WebGL Visualizer
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { E8RadarOverlay } from './e8_radar_overlay.js';

export class E8Visualizer3D {
    constructor() {
        this.canvas = document.getElementById('e8-canvas');
        if (!this.canvas) return;

        this.activeIndices = new Set();
        this.nodes = [];
        this.links = [];
        this.initE8Data();
        this.initThree();
        this.createScene();
        this.bindEvents();
        
        const wrapper = this.canvas.parentElement;
        if (wrapper) {
            this.radar = new E8RadarOverlay(wrapper, this);
        }
        
        this.animate();
    }

    initE8Data() {
        // Generates the 240 root vectors of E8 root system
        const roots = [];
        
        // 1. Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0)
        for (let i = 0; i < 8; i++) {
            for (let j = i + 1; j < 8; j++) {
                for (const s1 of [-1, 1]) {
                    for (const s2 of [-1, 1]) {
                        const v = new Array(8).fill(0);
                        v[i] = s1;
                        v[j] = s2;
                        roots.push(v);
                    }
                }
            }
        }
        
        // 2. (+-1/2, ..., +-1/2) with an even number of minus signs
        for (let bits = 0; bits < 256; bits++) {
            const signs = [];
            let negatives = 0;
            for (let i = 0; i < 8; i++) {
                const s = (bits & (1 << i)) !== 0 ? 1 : -1;
                signs.push(s);
                if (s === -1) negatives++;
            }
            if (negatives % 2 === 0) {
                const v = signs.map(s => s * 0.5);
                roots.push(v);
            }
        }

        // Project 8D coordinates to 3D via Coxeter-Icosian matrix math
        const phi = (1.0 + Math.sqrt(5.0)) / 2.0;
        const scale = 1.0 / Math.sqrt(1.0 + phi * phi);
        
        this.e8CoordinatesList = roots.map((v, idx) => {
            const x = (v[0] * phi + v[4]) * scale;
            const y = (v[1] * phi + v[5]) * scale;
            const z = (v[2] * phi + v[6]) * scale;
            const norm = Math.sqrt(x*x + y*y + z*z);
            return { pt3d: [x, y, z], original: v, idx: idx, norm: norm };
        });

        // Sort by radius for concentric shell grouping
        this.e8CoordinatesList.sort((a, b) => a.norm - b.norm);

        // Group into concentric shells [2, 30, 64, 64, 80]
        this.e8CoordinatesList.forEach((node, index) => {
            let shIdx = 0;
            if (index < 2) shIdx = 0;
            else if (index < 32) shIdx = 1;
            else if (index < 96) shIdx = 2;
            else if (index < 160) shIdx = 3;
            else shIdx = 4;
            
            node.shellIdx = shIdx;
        });
    }

    initThree() {
        const rect = this.canvas.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.scene = new THREE.Scene();
        
        // Skeuomorphic glass/holographic camera depth
        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 100);
        this.camera.position.set(0, 0, 8);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Analog radar grids
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 25;
        this.controls.minDistance = 2;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight1.position.set(5, 5, 5);
        this.scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-5, -5, -5);
        this.scene.add(dirLight2);

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
    }

    createScene() {
        this.latticeGroup = new THREE.Group();
        this.scene.add(this.latticeGroup);

        // Shell Colors (Amber, Emerald Green, Electric Cyan, Hot Orange, Crimson Red)
        const shellColors = [
            0xffb000, // Shell 0: Amber
            0x00bc50, // Shell 1: Emerald Green
            0x00f3ff, // Shell 2: Electric Cyan
            0xff6c00, // Shell 3: Hot Orange
            0xff004f  // Shell 4: Crimson Red
        ];

        // 1. Create nodes
        const sphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
        this.e8CoordinatesList.forEach(item => {
            const [x, y, z] = item.pt3d;
            const color = shellColors[item.shellIdx];
            
            // Skeuomorphic glossy bulb material
            const mat = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.15,
                shininess: 90,
                specular: 0xffffff
            });

            const mesh = new THREE.Mesh(sphereGeo, mat);
            mesh.position.set(x * 1.5, y * 1.5, z * 1.5);
            mesh.userData = item;
            
            this.latticeGroup.add(mesh);
            this.nodes.push(mesh);
        });

        // 2. Create connections (links between closest neighbours in Projected space)
        const linkMat = new THREE.LineBasicMaterial({
            color: 0x5a626f,
            transparent: true,
            opacity: 0.12
        });

        // Draw connections for nodes within 1.25 units distance
        const points = this.nodes.map(n => n.position);
        for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
                const dist = points[i].distanceTo(points[j]);
                if (dist < 1.25) {
                    const lineGeo = new THREE.BufferGeometry().setFromPoints([points[i], points[j]]);
                    const line = new THREE.Line(lineGeo, linkMat);
                    this.latticeGroup.add(line);
                }
            }
        }

        // 3. Add holographic radar rings (skeuomorphic analog reference)
        const ringMat = new THREE.LineBasicMaterial({ color: 0x444a53, transparent: true, opacity: 0.3 });
        for (let r = 1; r <= 3; r++) {
            const ringGeo = new THREE.RingGeometry(r * 1.1, r * 1.1 + 0.02, 64);
            const ring = new THREE.LineLoop(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            this.latticeGroup.add(ring);
        }
    }

    bindEvents() {
        window.addEventListener('resize', () => this.handleResize());
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        
        // HUD elements
        this.hud = document.getElementById('e8-hud');
        this.hudRoot = document.getElementById('hud-root');
        this.hudProj = document.getElementById('hud-proj');
        this.hudShell = document.getElementById('hud-shell');
        this.hudStatus = document.getElementById('hud-status');
    }

    handleResize() {
        const rect = this.canvas.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(this.width, this.height);
    }

    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / this.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / this.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.nodes);

        if (intersects.length > 0) {
            const node = intersects[0].object;
            const data = node.userData;
            
            // Show custom HUD (vintage style tooltip)
            if (this.hud) {
                this.hud.style.left = `${e.clientX - rect.left + 15}px`;
                this.hud.style.top = `${e.clientY - rect.top + 15}px`;
                this.hud.classList.remove('hidden');

                if (this.hudRoot) this.hudRoot.textContent = data.original.map(n => n.toFixed(1)).join(", ");
                if (this.hudProj) this.hudProj.textContent = data.pt3d.map(n => n.toFixed(3)).join(", ");
                if (this.hudShell) this.hudShell.textContent = `Shell ${data.shellIdx} (norm ${data.norm.toFixed(2)})`;
                
                const active = this.activeIndices.has(data.idx);
                if (this.hudStatus) {
                    this.hudStatus.textContent = active ? "ACTIVE PAGE (VRAM)" : "STANDBY (CPU PAGES)";
                    this.hudStatus.style.color = active ? "#00ff88" : "#7e839f";
                }
            }
        } else {
            if (this.hud) {
                this.hud.classList.add('hidden');
            }
        }
    }

    updateActiveCoordinates(activeSet) {
        this.activeIndices = activeSet;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Slow radar-like rotation
        if (this.latticeGroup) {
            this.latticeGroup.rotation.y += 0.003;
            this.latticeGroup.rotation.x += 0.001;
        }

        // Animate node scales and glowing intensities
        const time = Date.now() * 0.005;
        this.nodes.forEach(node => {
            const data = node.userData;
            const active = this.activeIndices.has(data.idx);
            
            if (active) {
                // Pulsate active nodes
                const scale = 1.6 + Math.sin(time) * 0.3;
                node.scale.set(scale, scale, scale);
                node.material.emissiveIntensity = 0.8 + Math.sin(time) * 0.4;
            } else {
                node.scale.set(1.0, 1.0, 1.0);
                node.material.emissiveIntensity = 0.15;
            }
        });

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
