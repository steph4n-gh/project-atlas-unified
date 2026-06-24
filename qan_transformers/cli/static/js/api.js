/**
 * GossetGate: API helper module
 */

export async function syncConfig(payload) {
    const res = await fetch('/api/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!res.ok) {
        throw new Error(`Config sync failed: ${res.statusText}`);
    }
    return res.json();
}

export async function loadModelStream(payload, onStep, onComplete, onError) {
    try {
        const response = await fetch('/api/model/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.body) {
            throw new Error("No readable body stream returned by model initialization API.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = JSON.parse(line.substring(6));
                    if (data.type === 'step') {
                        onStep(data);
                    } else if (data.type === 'complete') {
                        onComplete(data);
                    } else if (data.type === 'error') {
                        onError(data);
                    }
                }
            }
        }
    } catch (err) {
        onError({
            message: err.message || "Unknown stream connection failure.",
            traceback: ""
        });
    }
}

export async function ingestCodebase(folderPath) {
    const res = await fetch('/api/context/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ folder: folderPath })
    });
    if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Ingestion failed");
    }
    return res.json();
}

export async function runSelfImprove(payload) {
    const res = await fetch('/api/self-improve/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!res.ok) {
        throw new Error("Failed starting self-improvement loop");
    }
    return res.json();
}

export async function getSelfImproveStatus() {
    const res = await fetch('/api/self-improve/status');
    if (!res.ok) {
        throw new Error("Failed fetching self-improvement status");
    }
    return res.json();
}

export async function fetchContextFiles() {
    const res = await fetch('/api/context/files');
    if (!res.ok) {
        throw new Error("Failed fetching codebase files list");
    }
    return res.json();
}
