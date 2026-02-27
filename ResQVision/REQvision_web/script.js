// script.js
// REQvision Web Dashboard Map Logic

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Map
    // Default location: Vijayawada, India (16.5062, 80.6480)
    const map = L.map('map', {
        zoomControl: false // Hide default to place custom if needed, or leave default
    }).setView([16.5062, 80.6480], 13);
    
    // Move zoom control to bottom right so it doesn't overlap the command panel
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // 2. Define Tile Layers (Free / Open Source)
    
    // Normal View - OpenStreetMap
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    });

    // Satellite View - Esri World Imagery (Free to use for mapping)
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });

    // Default to Normal View
    osmLayer.addTo(map);

    // 3. Define Heatmap Layer
    // Heatmap data format: [lat, lng, risk_score]
    let heatmapData = [];
    
    const heatLayer = L.heatLayer(heatmapData, {
        radius: 25,
        blur: 15,
        maxZoom: 15,
        max: 1.0, // Maximum risk score is 1.0
        gradient: {
            0.2: 'blue', 
            0.4: 'cyan', 
            0.6: 'lime', 
            0.8: 'yellow', 
            1.0: 'red'
        }
    });

    // Heatmap defaults to ON
    heatLayer.addTo(map);

    // 4. Marker Layer for Points of Interest (Popups)
    const markerLayerGroup = L.layerGroup().addTo(map);

    /**
     * Add a marker to the map
     * @param {number} lat 
     * @param {number} lng 
     * @param {string} name 
     * @param {number} crowdCount 
     * @param {number} riskScore (0.0 to 1.0)
     */
    function addMarker(lat, lng, name, crowdCount, riskScore) {
        let riskLevel = "Low";
        let badgeClass = "low";
        
        if (riskScore >= 0.8) {
            riskLevel = "High";
            badgeClass = "high";
        } else if (riskScore >= 0.4) {
            riskLevel = "Medium";
            badgeClass = "medium";
        }

        const popupContent = `
            <div class="custom-popup">
                <h3>📍 ${name}</h3>
                <p>Crowd Count: <strong>${crowdCount}</strong></p>
                <p>Risk Score: <strong>${(riskScore * 100).toFixed(1)}%</strong></p>
                <p>Risk Level: <span class="badge ${badgeClass}">${riskLevel}</span></p>
            </div>
        `;

        const marker = L.marker([lat, lng]).bindPopup(popupContent);
        markerLayerGroup.addLayer(marker);
    }

    // 5. UI Controls Logic
    const btnNormal = document.getElementById('btn-normal');
    const btnSatellite = document.getElementById('btn-satellite');
    const btnHeatmap = document.getElementById('btn-heatmap');
    
    // Normal View Toggle
    btnNormal.addEventListener('click', () => {
        if (!map.hasLayer(osmLayer)) {
            map.removeLayer(satelliteLayer);
            osmLayer.addTo(map);
            btnNormal.classList.add('active');
            btnSatellite.classList.remove('active');
        }
    });

    // Satellite View Toggle
    btnSatellite.addEventListener('click', () => {
        if (!map.hasLayer(satelliteLayer)) {
            map.removeLayer(osmLayer);
            satelliteLayer.addTo(map);
            btnSatellite.classList.add('active');
            btnNormal.classList.remove('active');
        }
    });

    // Heatmap Toggle
    btnHeatmap.addEventListener('click', () => {
        if (map.hasLayer(heatLayer)) {
            map.removeLayer(heatLayer);
            btnHeatmap.classList.remove('active');
            btnHeatmap.innerHTML = '<span class="icon">🔥</span> Heatmap OFF';
        } else {
            heatLayer.addTo(map);
            btnHeatmap.classList.add('active');
            btnHeatmap.innerHTML = '<span class="icon">🔥</span> Heatmap ON';
        }
    });

    // 6. Dynamic Backend API Simulation (Bonus Feature)
    const alertBox = document.getElementById('alert-box');
    const alertMessage = document.getElementById('alert-message');

    function checkAlerts(data) {
        const criticalPoints = data.filter(pt => pt.riskScore > 0.8);
        if (criticalPoints.length > 0) {
            alertBox.classList.remove('hidden');
            alertMessage.innerText = `High risk (> 0.8) detected at ${criticalPoints.length} location(s)!`;
        } else {
            alertBox.classList.add('hidden');
        }
    }

    function updateDashboardData(newData) {
        // Clear old data
        heatmapData = [];
        markerLayerGroup.clearLayers();

        newData.forEach(point => {
            // Push to heatmap array: [lat, lng, intensity]
            heatmapData.push([point.lat, point.lng, point.riskScore]);
            
            // Add Marker
            addMarker(point.lat, point.lng, point.name, point.crowd, point.riskScore);
        });

        // Update heatmap layer
        heatLayer.setLatLngs(heatmapData);

        // Check for alerts
        checkAlerts(newData);
    }

    // Simulated API Polling every 5 seconds
    setInterval(() => {
        // Generate random dynamic data around Vijayawada for demonstration
        const mockApiData = [
            {
                name: "PNS Bus Station",
                lat: 16.5062 + (Math.random() - 0.5) * 0.02,
                lng: 80.6480 + (Math.random() - 0.5) * 0.02,
                crowd: Math.floor(Math.random() * 500) + 50,
                riskScore: Math.random() // Over 0.8 = Critical
            },
            {
                name: "Bhavani Island Ghat",
                lat: 16.5160 + (Math.random() - 0.5) * 0.01,
                lng: 80.6050 + (Math.random() - 0.5) * 0.01,
                crowd: Math.floor(Math.random() * 300) + 20,
                riskScore: Math.random() * 0.7 // Usually safe
            },
            {
                name: "Benz Circle",
                lat: 16.4971 + (Math.random() - 0.5) * 0.01,
                lng: 80.6485 + (Math.random() - 0.5) * 0.01,
                crowd: Math.floor(Math.random() * 1000) + 200,
                riskScore: Math.random() > 0.7 ? 0.85 : Math.random() * 0.5 // Occasional critical traffic
            }
        ];

        updateDashboardData(mockApiData);
        // console.log("Map data updated via API simulation.");
    }, 5000);

    // Initial Load trigger
    setTimeout(() => {
        updateDashboardData([
            { name: "PNS Bus Station", lat: 16.5062, lng: 80.6480, crowd: 250, riskScore: 0.6 },
            { name: "Bhavani Island Ghat", lat: 16.5160, lng: 80.6050, crowd: 100, riskScore: 0.2 },
            { name: "Benz Circle", lat: 16.4971, lng: 80.6485, crowd: 850, riskScore: 0.9 }
        ]);
    }, 500);
});
