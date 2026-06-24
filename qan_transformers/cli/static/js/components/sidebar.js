/**
 * GossetGate: Sidebar controls component
 */
import { syncConfig } from '../api.js';

export class SidebarPanel {
    constructor(appendLog) {
        this.appendLog = appendLog;
        this.initDOMElements();
        this.bindEvents();
        this.syncAll();
    }

    initDOMElements() {
        this.panel = document.getElementById('sidebar-panel');
        this.toggleBtn = document.getElementById('toggle-sidebar-btn');
        this.draftStrategy = document.getElementById('draft-strategy');
        this.draftModelGroup = document.getElementById('draft-model-group');
        
        // Config Sliders & Values
        this.sparseRatioSlider = document.getElementById('sparse-ratio-slider');
        this.sparseRatioVal = document.getElementById('sparse-ratio-val');
        this.firewallThresholdSlider = document.getElementById('firewall-threshold-slider');
        this.firewallThresholdVal = document.getElementById('firewall-threshold-val');
        
        // Toggles
        this.firewallToggle = document.getElementById('firewall-toggle');
        this.reviewToggle = document.getElementById('review-toggle');
        this.thinkingToggle = document.getElementById('thinking-toggle');
        this.telemetryToggle = document.getElementById('telemetry-toggle');
        
        // Max tokens
        this.maxTokensSlider = document.getElementById('max-tokens-slider');
        this.maxTokensVal = document.getElementById('max-tokens-val');
        
        // Codebase Ingestion path
        this.contextInput = document.getElementById('context-input');
    }

    bindEvents() {
        // Toggle Sidebar visibility
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

        // Accordion functionality for collapsible cards
        document.addEventListener('click', (e) => {
            const header = e.target.closest('.collapsible-card .card-header');
            if (!header) return;
            if (e.target.closest('button') || e.target.closest('a') || e.target.closest('input') || e.target.closest('select')) {
                return;
            }
            const card = header.closest('.collapsible-card');
            if (card) {
                card.classList.toggle('collapsed');
            }
        });

        // Speculative decoding select toggle
        if (this.draftStrategy) {
            this.draftStrategy.addEventListener('change', () => {
                if (this.draftStrategy.value === 'single') {
                    this.draftModelGroup.classList.add('hidden');
                } else {
                    this.draftModelGroup.classList.remove('hidden');
                }
            });
        }

        // Sliders & Toggles event listeners
        const addSliderListener = (slider, valDisplay, decimals = 2) => {
            if (slider && valDisplay) {
                slider.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    valDisplay.textContent = decimals === 0 ? parseInt(val) : val.toFixed(decimals);
                });
                slider.addEventListener('change', () => this.syncAll());
            }
        };

        addSliderListener(this.sparseRatioSlider, this.sparseRatioVal, 2);
        addSliderListener(this.firewallThresholdSlider, this.firewallThresholdVal, 2);
        addSliderListener(this.maxTokensSlider, this.maxTokensVal, 0);

        const addToggleListener = (toggleBtn) => {
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => {
                    toggleBtn.classList.toggle('active');
                    this.syncAll();
                });
            }
        };

        addToggleListener(this.firewallToggle);
        addToggleListener(this.reviewToggle);
        addToggleListener(this.thinkingToggle);
        addToggleListener(this.telemetryToggle);
    }

    async syncAll() {
        const payload = {
            sparse_ratio: this.sparseRatioSlider ? parseFloat(this.sparseRatioSlider.value) : 0.15,
            firewall_enabled: this.firewallToggle ? this.firewallToggle.classList.contains('active') : true,
            review_mode: this.reviewToggle ? this.reviewToggle.classList.contains('active') : false,
            threshold: this.firewallThresholdSlider ? parseFloat(this.firewallThresholdSlider.value) : 1.5,
            thinking_mode: (this.thinkingToggle && this.thinkingToggle.classList.contains('active')) ? "thinking" : "direct",
            max_new_tokens: this.maxTokensSlider ? parseInt(this.maxTokensSlider.value) : 128,
            optimize_telemetry: this.telemetryToggle ? this.telemetryToggle.classList.contains('active') : false
        };

        try {
            await syncConfig(payload);
            if (this.appendLog) {
                this.appendLog(`Config synchronized: sparse=${payload.sparse_ratio}, firewall=${payload.firewall_enabled}, threshold=${payload.threshold}, thinking=${payload.thinking_mode}, max_tokens=${payload.max_new_tokens}`);
            }
        } catch (err) {
            console.error("Failed syncing configuration to backend", err);
        }
    }
}
