// Sensor Reading Update
function updateSensorData() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            // Temperature
            const tempElement = document.getElementById('temperature');
            tempElement.textContent = data.temperature !== undefined ? 
                `${data.temperature} °C` : '--.- °C';
            
            // Humidity
            const humElement = document.getElementById('humidity');
            humElement.textContent = data.humidity !== undefined ? 
                `${data.humidity} %` : '--.- %';
            
            // Error
            const errorElement = document.getElementById('error');
            errorElement.textContent = data.error || '';
            errorElement.style.display = data.error ? 'block' : 'none';
            
            // Timestamp
            document.getElementById('timestamp').textContent = data.timestamp;
        })
        .catch(error => {
            console.error('Error fetching sensor data:', error);
        });
}

// Open Beak Detection Update
async function updateOpenBeakStatus() {
    try {
        const res = await fetch("/status?_=" + new Date().getTime()); // prevent cache
        const data = await res.json();
        document.getElementById("status").innerText = "Open Beak: " + data.count;
    } catch (e) {
        console.error("Failed to fetch open beak count:", e);
        document.getElementById("status").innerText = "Open Beak: ?";
    }
}

// Set intervals
setInterval(updateSensorData, 2000);      // Sensor data every 2s
setInterval(updateOpenBeakStatus, 500);   // Beak status every 0.5s

// Initial calls
updateSensorData();
updateOpenBeakStatus();
