// ==========================================
// CONFIGURATION
// ==========================================
const API_URL = "http://127.0.0.1:8000/predict"; // TODO: Update to production URL (e.g. Render/Vercel)

// ==========================================
// DOM ELEMENTS
// ==========================================
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const loaderOverlay = document.getElementById('loader-overlay');

const howItWorks = document.getElementById('how-it-works');
const resultsView = document.getElementById('results-view');
const resetBtn = document.getElementById('reset-btn');

const primaryDiagnosis = document.getElementById('primary-diagnosis');
const primaryBar = document.getElementById('primary-bar');
const primaryPct = document.getElementById('primary-pct');

const actionCard = document.getElementById('action-card');
const actionIcon = document.getElementById('action-icon');
const actionTitle = document.getElementById('action-title');
const actionText = document.getElementById('action-text');

const secondaryMatches = document.getElementById('secondary-matches');
const toastContainer = document.getElementById('toast-container');
const sampleBtns = document.querySelectorAll('.sample-btn');

// ==========================================
// EVENT LISTENERS
// ==========================================

// Drag and drop mechanics
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => uploadZone.classList.add('drop-active'));
});

['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('drop-active'));
});

uploadZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));

// Reset Application State
resetBtn.addEventListener('click', () => {
    // Hide Results, Show How It Works
    resultsView.classList.replace('flex', 'hidden');
    resultsView.classList.remove('opacity-100');
    
    setTimeout(() => {
        howItWorks.classList.replace('hidden', 'block');
        setTimeout(() => howItWorks.classList.add('opacity-100'), 50);
    }, 300);

    // Hide Preview, Show Uploader
    previewContainer.classList.add('hidden');
    uploadZone.classList.replace('hidden', 'flex');
    
    fileInput.value = ''; 
    
    // Reset Progress Bars
    primaryBar.style.width = '0%';
});

// Sample Images Logic
sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
        const url = btn.getAttribute('data-url');
        showToast('Loading sample image...', 'info');
        
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            // Convert blob to File object to match file upload format
            const file = new File([blob], "sample.jpg", { type: "image/jpeg" });
            handleFiles([file]);
        } catch (error) {
            showToast('Failed to load sample image.', 'error');
        }
    });
});

// ==========================================
// CORE LOGIC
// ==========================================

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showToast('Please upload a valid image file (JPEG, PNG).', 'error');
        return;
    }

    // Prepare UI: Swap Upload Zone for Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadZone.classList.replace('flex', 'hidden');
        previewContainer.classList.remove('hidden');
        
        analyzeImage(file);
    };
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    // Enable Loading State
    imagePreview.classList.add('blur-preview');
    loaderOverlay.classList.replace('hidden', 'flex');
    
    // Hide old results, show loader side
    resultsView.classList.replace('flex', 'hidden');
    resultsView.classList.remove('opacity-100');
    howItWorks.classList.replace('block', 'hidden');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        
        const data = await response.json();
        if (data.predictions && data.predictions.length > 0) {
            populateResults(data.predictions);
        } else {
            throw new Error("Invalid response format.");
        }
    } catch (error) {
        console.error('API Error:', error);
        showToast('Analysis failed. Is the FastAPI server running?', 'error');
        
        // Soft reset UI
        setTimeout(() => {
            uploadZone.classList.replace('hidden', 'flex');
            previewContainer.classList.add('hidden');
            howItWorks.classList.replace('hidden', 'block');
        }, 2000);
    } finally {
        // Remove Loader
        imagePreview.classList.remove('blur-preview');
        loaderOverlay.classList.replace('flex', 'hidden');
    }
}

function populateResults(predictions) {
    const topPred = predictions[0];
    const percentage = (topPred.confidence * 100).toFixed(1);
    
    // Set Top Match
    primaryDiagnosis.innerText = topPred.class;
    primaryPct.innerText = `${percentage}%`;
    
    // Setup Action Card based on 'healthy' vs 'disease'
    const isHealthy = topPred.class.toLowerCase().includes('healthy');
    
    if (isHealthy) {
        actionCard.className = "rounded-2xl p-5 shadow-sm border border-brand-200 bg-brand-50 text-brand-900";
        actionIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>`;
        actionTitle.innerText = "Status: Healthy";
        actionText.innerText = "No immediate action needed. Continue current watering and nutrient regimen to maintain optimal plant health.";
    } else {
        actionCard.className = "rounded-2xl p-5 shadow-sm border border-red-200 bg-red-50 text-red-900";
        actionIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>`;
        actionTitle.innerText = "Suggested Treatment";
        actionText.innerText = "Isolate the affected plant immediately. Remove severely damaged leaves. Consider applying a targeted fungicide or bactericide depending on the specific pathogen.";
    }

    // Set Secondary Matches
    secondaryMatches.innerHTML = '';
    predictions.slice(1, 3).forEach((pred, index) => {
        const pct = (pred.confidence * 100).toFixed(1);
        const colorClass = index === 0 ? 'bg-blue-400' : 'bg-gray-400';
        
        secondaryMatches.insertAdjacentHTML('beforeend', `
            <div>
                <div class="flex justify-between items-end mb-1">
                    <span class="text-sm font-medium text-gray-700">${pred.class}</span>
                    <span class="text-xs font-bold text-gray-500">${pct}%</span>
                </div>
                <div class="w-full bg-gray-100 rounded-full h-1.5 overflow-hidden">
                    <div class="progress-bar-fill h-1.5 rounded-full ${colorClass}" style="width: 0%" data-target="${pct}"></div>
                </div>
            </div>
        `);
    });

    // Reveal Results View
    howItWorks.classList.replace('block', 'hidden');
    howItWorks.classList.remove('opacity-100');
    
    resultsView.classList.replace('hidden', 'flex');
    // Small delay to allow display block to render before opacity transition
    setTimeout(() => {
        resultsView.classList.add('opacity-100');
        
        // Trigger Progress Bar Animations
        primaryBar.style.width = `${percentage}%`;
        
        document.querySelectorAll('#secondary-matches .progress-bar-fill').forEach(bar => {
            bar.style.width = `${bar.getAttribute('data-target')}%`;
        });
    }, 50);
}

// ==========================================
// TOAST NOTIFICATIONS
// ==========================================
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    
    let bgColor = 'bg-gray-800';
    let icon = `<svg class="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    
    if (type === 'error') {
        bgColor = 'bg-red-600';
        icon = `<svg class="w-5 h-5 text-red-100" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    }

    toast.className = `toast-enter flex items-center p-4 rounded-xl shadow-xl ${bgColor}`;
    toast.innerHTML = `
        <div class="mr-3">${icon}</div>
        <div class="text-sm font-medium text-white">${message}</div>
    `;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.replace('toast-enter', 'toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
