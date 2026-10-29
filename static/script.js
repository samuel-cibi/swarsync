let selectedFile = null;
let refFile = null;
let userFile = null;

function switchMode(mode) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    if (mode === 'learn') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('mode-learn').style.display = 'block';
        document.getElementById('mode-sing').style.display = 'none';
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('mode-learn').style.display = 'none';
        document.getElementById('mode-sing').style.display = 'block';
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        document.getElementById('file-label').innerText = `Selected: ${file.name}`;
        document.getElementById('analyze-btn').removeAttribute('disabled');
    }
}

function handleRefSelect(event) {
    const file = event.target.files[0];
    if (file) {
        refFile = file;
        document.getElementById('ref-label').innerText = `Ref: ${file.name}`;
        checkSingingReady();
    }
}

function handleUserSelect(event) {
    const file = event.target.files[0];
    if (file) {
        userFile = file;
        document.getElementById('user-label').innerText = `User: ${file.name}`;
        checkSingingReady();
    }
}

function checkSingingReady() {
    if (refFile && userFile) {
        document.getElementById('evaluate-btn').removeAttribute('disabled');
    }
}

async function analyzeSong() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('audio', selectedFile);

    document.getElementById('loader').style.display = 'block';
    document.getElementById('results-container').style.display = 'none';
    document.getElementById('analyze-btn').setAttribute('disabled', 'true');

    try {
        const response = await fetch('/api/analyse-song', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            document.getElementById('res-tempo').innerText = data.tempo_bpm;
            document.getElementById('res-duration').innerText = data.duration_seconds;
            document.getElementById('res-beats').innerText = data.beats_count;

            if (data.time_signature) {
                document.getElementById('res-ts').innerText = data.time_signature.label;
                document.getElementById('res-kind').innerText = `${data.time_signature.kind} meter`;
                document.getElementById('res-details').innerText = `Beats per bar: ${data.time_signature.beats_per_bar} | Beat BPM: ${data.time_signature.beat_bpm} | Grouping: ${data.time_signature.grouping}`;
            } else {
                document.getElementById('res-ts').innerText = "N/A";
                document.getElementById('res-kind').innerText = "Rhythm unclear";
                document.getElementById('res-details').innerText = "Could not estimate time signature.";
            }

            document.getElementById('res-timestamps').innerText = data.beat_times.slice(0, 10).join(', ') + ' ...';
            document.getElementById('results-container').style.display = 'block';
        } else {
            alert('Analysis Error: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Failed to connect: ' + err.message);
    } finally {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('analyze-btn').removeAttribute('disabled');
    }
}

async function evaluateSinging() {
    if (!refFile || !userFile) return;

    const formData = new FormData();
    formData.append('reference', refFile);
    formData.append('user_recording', userFile);

    document.getElementById('singing-loader').style.display = 'block';
    document.getElementById('singing-results').style.display = 'none';
    document.getElementById('evaluate-btn').setAttribute('disabled', 'true');

    try {
        const response = await fetch('/api/evaluate-singing', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            document.getElementById('v-score').innerText = data.overall_score;
            document.getElementById('v-error').innerText = data.mean_intonation_error_cents;
            document.getElementById('v-stable').innerText = data.stable_voiced_percentage;
            document.getElementById('v-sd').innerText = `${data.pitch_sd} Hz`;
            document.getElementById('v-timing').innerText = data.timing_classification;
            document.getElementById('v-vibrato').innerText = `${data.vibrato_rate_hz} Hz (${data.vibrato_breadth_cents} cents)`;
            document.getElementById('v-feedback').innerText = data.feedback;
            document.getElementById('singing-results').style.display = 'block';
        } else {
            alert('Evaluation Error: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Failed to connect: ' + err.message);
    } finally {
        document.getElementById('singing-loader').style.display = 'none';
        document.getElementById('evaluate-btn').removeAttribute('disabled');
    }
}