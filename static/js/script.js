const fileInput = document.getElementById('file-input');
const dropArea = document.getElementById('drop-area');
const form = document.getElementById('upload-form');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('result-container');
const resultText = document.getElementById('result-text');
const accuracyText = document.getElementById('accuracy-text');
const previewImg = document.getElementById('image-preview');

// Tabs logic
const uploadTab = document.getElementById('upload-tab');
const realtimeTab = document.getElementById('realtime-tab');
const uploadCard = document.getElementById('upload-card');
const realtimeCard = document.getElementById('realtime-card');
const video = document.getElementById('webcam');
const rtResult = document.getElementById('realtime-result-text');

let isRealtimeActive = false;
let stream = null;

uploadTab.addEventListener('click', () => {
    switchTab('upload');
});

realtimeTab.addEventListener('click', () => {
    switchTab('realtime');
});

function switchTab(mode) {
    if (mode === 'upload') {
        uploadTab.classList.add('active');
        realtimeTab.classList.remove('active');
        uploadCard.classList.remove('hidden');
        realtimeCard.classList.add('hidden');
        stopWebcam();
    } else {
        realtimeTab.classList.add('active');
        uploadTab.classList.remove('active');
        realtimeCard.classList.remove('hidden');
        uploadCard.classList.add('hidden');
        startWebcam();
    }
}

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        isRealtimeActive = true;
        processRealtimeFrames();
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        rtResult.innerText = "Camera access denied or not available.";
    }
}

function stopWebcam() {
    isRealtimeActive = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

async function processRealtimeFrames() {
    if (!isRealtimeActive) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 300;
    canvas.height = video.videoHeight || 300;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg', 0.7);

    try {
        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();

        if (data.result) {
            rtResult.innerText = `Detected: ${data.result}`;
            rtResult.style.color = data.result === 'Defective' ? '#db4437' : '#0f9d58';
        }
    } catch (err) {
        console.error("Realtime error: ", err);
    }

    // Delay next frame to save resources
    setTimeout(processRealtimeFrames, 500);
}

// Auto-submit when image is selected
fileInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        await handleUpload(e.target.files[0]);
    }
});

// Drag and drop support
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = '#e8f0fe';
    dropArea.style.borderColor = '#4285f4';
});

dropArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = 'transparent';
    dropArea.style.borderColor = '#dadce0';
});

dropArea.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = 'transparent';
    if (e.dataTransfer.files.length > 0) {
        await handleUpload(e.dataTransfer.files[0]);
    }
});

async function handleUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    form.classList.add('hidden');
    loading.classList.remove('hidden');
    resultContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        loading.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        if (data.error) {
            resultText.innerText = 'Error';
            resultText.style.color = '#db4437';
            accuracyText.innerText = data.error;
        } else {
            resultText.innerText = `Result: ${data.result}`;
            resultText.style.color = data.result === 'Defective' ? '#db4437' : '#0f9d58';
            accuracyText.innerText = ``;
        }
    } catch (error) {
        loading.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        resultText.innerText = 'Error';
        accuracyText.innerText = 'Server could not process image.';
    }
}

function resetForm() {
    form.classList.remove('hidden');
    loading.classList.add('hidden');
    resultContainer.classList.add('hidden');
    fileInput.value = '';
}
