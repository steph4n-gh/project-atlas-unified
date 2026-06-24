/**
 * GossetGate: Audit Component (Cohomology AST Graph & Self-Improvement Loop)
 */

export class AuditPanel {
    constructor(appendLog) {
        this.appendLog = appendLog;
        this.siEventSource = null;
        this.initDOMElements();
        this.bindEvents();
        this.resizeCanvas();
    }

    initDOMElements() {
        this.auditCodeInput = document.getElementById('audit-code-input');
        this.auditTauSlider = document.getElementById('audit-tau-slider');
        this.auditTauVal = document.getElementById('audit-tau-val');
        this.runAuditBtn = document.getElementById('run-audit-btn');
        this.auditResultsContainer = document.getElementById('audit-results-container');
        this.auditStatusBadge = document.getElementById('audit-status-badge');
        this.auditLambda2Val = document.getElementById('audit-lambda2-val');
        this.auditMsg = document.getElementById('audit-msg');
        
        this.auditCanvas = document.getElementById('ast-graph-canvas');
        if (this.auditCanvas) {
            this.auditCtx = this.auditCanvas.getContext('2d');
        }

        // Self-Improvement DOM
        this.siBackendSelect = document.getElementById('si-backend-select');
        this.siTargetSelect = document.getElementById('si-target-select');
        this.siGenerationsInput = document.getElementById('si-generations-input');
        this.startSiBtn = document.getElementById('start-si-btn');
        this.siStatusIndicator = document.getElementById('si-status-indicator');
        this.siLogConsole = document.getElementById('si-log-console');
    }

    bindEvents() {
        if (this.auditTauSlider && this.auditTauVal) {
            this.auditTauSlider.addEventListener('input', (e) => {
                this.auditTauVal.textContent = parseFloat(e.target.value).toFixed(2);
            });
        }

        if (this.runAuditBtn) {
            this.runAuditBtn.addEventListener('click', () => this.runConnectivityAudit());
        }

        if (this.startSiBtn) {
            this.startSiBtn.addEventListener('click', () => this.startSelfImprovement());
        }

        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        if (!this.auditCanvas) return;
        const container = this.auditCanvas.parentElement;
        this.auditCanvas.width = container.clientWidth;
        this.auditCanvas.height = container.clientHeight || 300;
        this.drawAuditGraph([], [], []);
    }

    async runConnectivityAudit() {
        const code = this.auditCodeInput.value.trim();
        const tau = parseFloat(this.auditTauSlider.value);
        if (!code) {
            alert("Please paste some Python source code first.");
            return;
        }

        this.runAuditBtn.disabled = true;
        this.runAuditBtn.textContent = "Analyzing...";
        this.appendLog("Sending AST code audit request to cohomology validator...");

        try {
            const res = await fetch('/api/audit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, tau })
            });
            
            if (res.ok) {
                const data = await res.json();
                
                if (this.auditResultsContainer) {
                    this.auditResultsContainer.classList.remove('hidden');
                }
                
                if (this.auditStatusBadge) {
                    if (data.approved) {
                        this.auditStatusBadge.textContent = "Approved";
                        this.auditStatusBadge.className = "audit-status-badge approved";
                    } else {
                        this.auditStatusBadge.textContent = "Fractured";
                        this.auditStatusBadge.className = "audit-status-badge fractured";
                    }
                }
                
                if (this.auditLambda2Val) {
                    this.auditLambda2Val.textContent = data.lambda_2.toFixed(4);
                }
                if (this.auditMsg) {
                    this.auditMsg.textContent = data.message;
                }
                
                this.appendLog(`Audit complete: approved=${data.approved}, lambda2=${data.lambda_2.toFixed(4)}`);
                this.drawAuditGraph(data.nodes, data.edges, data.undefined_nodes || []);
            } else {
                alert("Failed to analyze connectivity.");
            }
        } catch (err) {
            console.error(err);
            alert("Error during audit execution.");
        } finally {
            this.runAuditBtn.disabled = false;
            this.runAuditBtn.textContent = "Analyze Connectivity";
        }
    }

    drawAuditGraph(nodes, edges, undefinedNodes) {
        if (!this.auditCtx || !this.auditCanvas) return;
        const W = this.auditCanvas.width;
        const H = this.auditCanvas.height;
        this.auditCtx.clearRect(0, 0, W, H);

        if (!nodes || nodes.length === 0) {
            this.auditCtx.fillStyle = "rgba(255, 255, 255, 0.3)";
            this.auditCtx.font = "14px 'Outfit', sans-serif";
            this.auditCtx.textAlign = "center";
            this.auditCtx.fillText("No graph computed. Paste code and analyze to visualize AST structure.", W / 2, H / 2);
            return;
        }

        const cx = W / 2;
        const cy = H / 2;
        const radius = Math.min(W, H) * 0.35;
        const nodePositions = {};

        nodes.forEach((node, idx) => {
            const angle = (idx * 2 * Math.PI) / nodes.length;
            nodePositions[node] = {
                x: cx + radius * Math.cos(angle),
                y: cy + radius * Math.sin(angle)
            };
        });

        // Draw edges
        this.auditCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        this.auditCtx.lineWidth = 1.5;
        edges.forEach(([src, dst]) => {
            const p1 = nodePositions[src];
            const p2 = nodePositions[dst];
            if (p1 && p2) {
                this.auditCtx.beginPath();
                this.auditCtx.moveTo(p1.x, p1.y);
                this.auditCtx.lineTo(p2.x, p2.y);
                this.auditCtx.stroke();
            }
        });

        // Draw nodes
        nodes.forEach((node) => {
            const pos = nodePositions[node];
            const isUndefined = undefinedNodes.includes(node);
            
            // Outer glow
            this.auditCtx.shadowBlur = 10;
            this.auditCtx.shadowColor = isUndefined ? "rgba(255, 75, 75, 0.8)" : "rgba(0, 242, 254, 0.8)";
            
            this.auditCtx.fillStyle = isUndefined ? "rgba(255, 75, 75, 0.9)" : "rgba(0, 242, 254, 0.9)";
            this.auditCtx.beginPath();
            this.auditCtx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI);
            this.auditCtx.fill();
            
            // Label
            this.auditCtx.shadowBlur = 0;
            this.auditCtx.fillStyle = "rgba(255, 255, 255, 0.8)";
            this.auditCtx.font = "10px 'Fira Code', monospace";
            this.auditCtx.textAlign = "center";
            this.auditCtx.fillText(node, pos.x, pos.y - 10);
        });
    }

    async startSelfImprovement() {
        const backend = this.siBackendSelect.value;
        const target = this.siTargetSelect.value;
        const generations = parseInt(this.siGenerationsInput.value) || 3;

        this.startSiBtn.disabled = true;
        this.startSiBtn.textContent = "Starting...";
        this.siLogConsole.textContent = "Connecting to background execution thread...\n";
        this.siStatusIndicator.textContent = "Status: Initializing";
        this.siStatusIndicator.style.color = "#ffb000";
        
        try {
            const url = `/api/self-improve/start?backend=${backend}&generations=${generations}&target=${target}`;
            const res = await fetch(url, { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started' || data.status === 'already_running') {
                this.siStatusIndicator.textContent = "Status: Executing";
                this.siStatusIndicator.style.color = "#00f3ff";
                this.startSiBtn.textContent = "Running Loop";
                
                if (this.siEventSource) {
                    this.siEventSource.close();
                }
                
                this.siEventSource = new EventSource('/api/self-improve/stream');
                this.siEventSource.onmessage = (e) => {
                    this.siLogConsole.textContent += `\n${e.data}`;
                    this.siLogConsole.scrollTop = this.siLogConsole.scrollHeight;
                    
                    if (e.data.includes("=== Self-Improvement Loop Complete ===")) {
                        this.siStatusIndicator.textContent = "Status: Complete";
                        this.siStatusIndicator.style.color = "#00ff88";
                        this.startSiBtn.disabled = false;
                        this.startSiBtn.textContent = "Start Self-Improvement Loop";
                        this.siEventSource.close();
                    }
                };
            } else {
                alert("Could not start loop.");
                this.startSiBtn.disabled = false;
                this.startSiBtn.textContent = "Start Self-Improvement Loop";
            }
        } catch (err) {
            console.error(err);
            alert("Error calling self-improve API.");
            this.startSiBtn.disabled = false;
            this.startSiBtn.textContent = "Start Self-Improvement Loop";
        }
    }
}
