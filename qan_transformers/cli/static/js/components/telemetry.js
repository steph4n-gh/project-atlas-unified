/**
 * GossetGate: Telemetry component
 */

export class TelemetryPanel {
    constructor() {
        this.initDOMElements();
        this.toggleBtn = document.getElementById('toggle-viz-btn');
        this.panel = document.getElementById('visualization-panel');
        
        if (this.toggleBtn && this.panel) {
            this.toggleBtn.addEventListener('click', () => {
                this.panel.classList.toggle('collapsed');
                if (this.panel.classList.contains('collapsed')) {
                    this.toggleBtn.classList.remove('active');
                } else {
                    this.toggleBtn.classList.add('active');
                }
            });
        }
    }

    initDOMElements() {
        this.metricSpeed = document.getElementById('metric-speed');
        this.metricVram = document.getElementById('metric-vram');
        this.metricCells = document.getElementById('metric-cells');
        this.metricL2 = document.getElementById('metric-l2');
        this.metricContext = document.getElementById('metric-context');
        this.metricAcceptance = document.getElementById('metric-acceptance');
        this.logConsole = document.getElementById('log-console');
    }

    update(data) {
        if (data.speed !== undefined && this.metricSpeed) {
            this.metricSpeed.textContent = data.speed.toFixed(1);
        }
        if (data.vram_saved !== undefined && this.metricVram) {
            this.metricVram.textContent = `${data.vram_saved.toFixed(0)}%`;
        }
        if (data.active_cells !== undefined && this.metricCells) {
            this.metricCells.textContent = data.active_cells;
        }
        if (data.lambda_2 !== undefined && this.metricL2) {
            const l2Val = parseFloat(data.lambda_2);
            this.metricL2.textContent = isNaN(l2Val) ? "---" : l2Val.toFixed(3);
            
            // Highlight fracture event if firewall detects obstruction
            if (data.is_fractured) {
                this.metricL2.style.color = "#ff3b6f";
                this.metricL2.style.textShadow = "0 0 8px #ff3b6f";
            } else {
                this.metricL2.style.color = "";
                this.metricL2.style.textShadow = "";
            }
        }
        if (data.context_size !== undefined && this.metricContext) {
            this.metricContext.textContent = data.context_size;
        }
        if (this.metricAcceptance) {
            if (data.speculative && data.acceptance_rate !== undefined) {
                this.metricAcceptance.textContent = `${data.acceptance_rate.toFixed(1)}%`;
            } else {
                this.metricAcceptance.textContent = "N/A";
            }
        }
    }

    appendLog(msg) {
        if (!this.logConsole) return;
        const now = new Date().toLocaleTimeString();
        this.logConsole.textContent += `\n[${now}] ${msg}`;
        // Cap log size to avoid unbounded memory leak in browser
        if (this.logConsole.textContent.length > 50000) {
            this.logConsole.textContent = this.logConsole.textContent.slice(-25000);
        }
        this.logConsole.scrollTop = this.logConsole.scrollHeight;
    }
}
