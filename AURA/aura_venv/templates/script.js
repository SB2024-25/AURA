document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('fileInput');
    const resultsContainer = document.getElementById('resultsContainer');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const file = fileInput.files[0];
        
        // Show loading state
        const submitBtn = uploadForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';

        try {
            if (!file) {
                throw new Error('Please select a file to upload');
            }
            
            // Validate file type
            const validExtensions = ['.csv', '.json', '.xml'];
            const fileExt = file.name.split('.').pop().toLowerCase();
            if (!validExtensions.includes('.' + fileExt)) {
                throw new Error('Invalid file type. Please upload a CSV, JSON, or XML file.');
            }

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            localStorage.setItem('currentFile', data.file);
            window.location.href = data.redirect;
        } catch (error) {
            alert(error.message);
        } finally {
            // Restore button state
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Upload';
        }
    });


    analyzeBtn.addEventListener('click', async function() {
        const currentFile = localStorage.getItem('currentFile');
        if (!currentFile) {
            alert('Please upload a file first');
            return;
        }

        try {
            // Show loading indicator
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';

            const description = document.getElementById('description').value;
            const expectations = document.getElementById('expectations').value;
            const outlierMethod = document.getElementById('outlierMethod').value;
            const outlierThreshold = document.getElementById('outlierThreshold').value;

            const response = await fetch('/insights', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file: currentFile,
                    description: description,
                    expectations: expectations,
                    outlier_method: outlierMethod,
                    outlier_threshold: outlierThreshold
                })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }

            localStorage.setItem('analysisResults', JSON.stringify(data));
            window.location.href = data.redirect;
        } catch (error) {
            alert(error.message);
        } finally {
            // Restore button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Data';
        }
    });


    // Show analysis results if available
    const results = JSON.parse(localStorage.getItem('analysisResults'));
    if (results) {
        displayResults(results);
    }
});

function displayResults(results) {
    const container = document.getElementById('resultsContainer');
    
    // Clear previous results
    container.innerHTML = '';

    // Display AI Insights
    const insightsDiv = document.createElement('div');
    insightsDiv.className = 'mb-4';
    insightsDiv.innerHTML = `
        <h5>AI Insights</h5>
        <div class="card">
            <div class="card-body">
                <p>${results.ai_insights.summary}</p>
            </div>
        </div>
    `;
    container.appendChild(insightsDiv);

    // Display Visualizations
    const vizDiv = document.createElement('div');
    vizDiv.className = 'mb-4';
    vizDiv.innerHTML = `
        <h5>Suggested Visualizations</h5>
        <div class="card">
            <div class="card-body">
                ${results.ai_insights.visualization_suggestions.map(viz => `
                    <div class="mb-3">
                        <p><strong>${viz.type}</strong>: ${viz.description}</p>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    container.appendChild(vizDiv);

    // Display Data Quality
    const qualityDiv = document.createElement('div');
    qualityDiv.className = 'mb-4';
    qualityDiv.innerHTML = `
        <h5>Data Quality Report</h5>
        <div class="card">
            <div class="card-body">
                <p>Missing Values: ${JSON.stringify(results.data_quality.missing_values)}</p>
                <p>Duplicates: ${results.data_quality.duplicates}</p>
                <p>Outliers: ${JSON.stringify(results.data_quality.outliers)}</p>
            </div>
        </div>
    `;
    container.appendChild(qualityDiv);
}
