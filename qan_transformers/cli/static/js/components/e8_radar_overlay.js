import * as THREE from 'three';

export class E8RadarOverlay {
    constructor(canvasWrapper, threeVisualizer) {
        this.wrapper = canvasWrapper;
        this.visualizer = threeVisualizer; // Reference to E8Visualizer3D instance
        
        // 1. Create overlay canvas element
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'e8-radar-overlay';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none'; // CRITICAL: Clicks pass to WebGL canvas below
        this.canvas.style.zIndex = '5';           // Positioned above Three.js canvas, below HUD
        
        this.wrapper.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }
    
    resize() {
        const rect = this.wrapper.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        // Handle high-DPI displays to prevent blurry canvas drawings
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const cx = this.width / 2;
        const cy = this.height / 2;
        const radius = Math.min(this.width, this.height) * 0.45;
        
        // Clear background with high transparency for soft trails
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // 1. Draw static analog grid rings (very faint, subtle)
        this.ctx.strokeStyle = 'rgba(0, 243, 255, 0.04)';
        this.ctx.lineWidth = 1;
        
        // Concentric circles
        for (let r = 0.25; r <= 1.00; r += 0.25) {
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
            this.ctx.stroke();
        }
        
        // Faint crosshairs
        this.ctx.beginPath();
        this.ctx.moveTo(cx - radius, cy);
        this.ctx.lineTo(cx + radius, cy);
        this.ctx.moveTo(cx, cy - radius);
        this.ctx.lineTo(cx, cy + radius);
        this.ctx.stroke();
        
        // 2. Draw steady breathing halos on screen projections of active VRAM page coordinates
        if (this.visualizer && this.visualizer.activeIndices && this.visualizer.nodes && this.visualizer.camera) {
            const tempV = new THREE.Vector3();
            const pulse = 1.0 + Math.sin(Date.now() * 0.005) * 0.15;
            
            this.visualizer.nodes.forEach(node => {
                const data = node.userData;
                if (!data) return;
                
                const active = this.visualizer.activeIndices.has(data.idx);
                if (active) {
                    // Project 3D node world coordinates to 2D Screen Space (-1 to +1 range)
                    node.getWorldPosition(tempV);
                    tempV.project(this.visualizer.camera);
                    
                    // Translate NDC to canvas pixel coordinates
                    const sx = (tempV.x * 0.5 + 0.5) * this.width;
                    const sy = (-tempV.y * 0.5 + 0.5) * this.height;
                    
                    const dx = sx - cx;
                    const dy = sy - cy;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    
                    // Draw if within visualizer boundary
                    if (dist <= radius) {
                        const color = node.material.color.getStyle(); // e.g. "rgb(0, 243, 255)"
                        
                        // Outer breathing ring
                        this.ctx.strokeStyle = color.replace('rgb', 'rgba').replace(')', `, 0.25)`);
                        this.ctx.lineWidth = 1;
                        this.ctx.beginPath();
                        this.ctx.arc(sx, sy, 5 + pulse * 4, 0, Math.PI * 2);
                        this.ctx.stroke();
                        
                        // Center active dot
                        this.ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', `, 0.7)`);
                        this.ctx.beginPath();
                        this.ctx.arc(sx, sy, 2, 0, Math.PI * 2);
                        this.ctx.fill();
                    }
                }
            });
        }
        
        // 3. Draw mathematical system telemetry HUD overlay (vintage green/cyan cockpit terminal style)
        this.ctx.font = '9px monospace';
        this.ctx.fillStyle = 'rgba(0, 243, 255, 0.65)';
        
        // Retrieve actual stats from the dashboard UI elements
        const speedVal = document.getElementById("metric-speed")?.textContent || "0.0";
        const vramVal = document.getElementById("metric-vram")?.textContent || "0.0%";
        const cellsVal = document.getElementById("metric-cells")?.textContent || "0";
        const l2Val = document.getElementById("metric-l2")?.textContent || "---";
        const contextVal = document.getElementById("metric-context")?.textContent || "0";
        const acceptanceVal = document.getElementById("metric-acceptance")?.textContent || "N/A";
        const currentModel = document.getElementById("status-model-lbl")?.textContent?.replace("Model: ", "") || "Offline";
        
        // Draw HUD lines in the upper-left corner
        this.ctx.fillText("--- QAN COCKPIT HUD METRICS ---", 12, 22);
        this.ctx.fillText(`MODEL_IDENTIFIER : ${currentModel}`, 12, 34);
        this.ctx.fillText(`ATTN_THROUGHPUT  : ${speedVal} tok/s`, 12, 46);
        this.ctx.fillText(`MORSE_KV_SAVED   : ${vramVal}`, 12, 58);
        this.ctx.fillText(`ACTIVE_E8_PAGES  : ${cellsVal} / 240`, 12, 70);
        this.ctx.fillText(`COHOMOLOGY_L2    : ${l2Val}`, 12, 82);
        this.ctx.fillText(`INGESTED_CONTEXT : ${contextVal} tokens`, 12, 94);
        this.ctx.fillText(`SPEC_ACCEPT_RATE : ${acceptanceVal}`, 12, 106);
    }
}

