// Dashboard JavaScript for Line Follower Robot
class RobotDashboard {
    constructor() {
        this.socket = io();
        this.pathCanvas = document.getElementById('path-canvas');
        this.pathCtx = this.pathCanvas.getContext('2d');
        this.metricsChart = null;
        this.pathHistory = [];
        this.metrics = {
            speed: 0,
            accuracy: 100,
            objectsDetected: 0,
            uptime: 0
        };
        
        this.initializeCharts();
        this.setupEventListeners();
        this.startAnimations();
    }

    initializeCharts() {
        // Initialize metrics chart
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        this.metricsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Line Accuracy',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Speed',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    setupEventListeners() {
        this.socket.on('robot_update', (data) => {
            this.updateCameraFeed(data.frame);
            this.updateStatus(data.data);
            this.updatePath(data.data.path_history);
            this.updateMetrics(data.data);
        });

        // Add keyboard controls
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e.key);
        });
    }

    updateCameraFeed(frameData) {
        const cameraFeed = document.getElementById('camera-feed');
        cameraFeed.src = 'data:image/jpeg;base64,' + frameData;
        
        // Add pulse effect when objects detected
        if (this.metrics.objectsDetected > 0) {
            cameraFeed.classList.add('pulse-red');
            setTimeout(() => cameraFeed.classList.remove('pulse-red'), 500);
        }
    }

    updateStatus(data) {
        // Line status with smooth transitions
        const lineStatus = document.getElementById('line-status');
        const isDetected = data.line_detected;
        
        lineStatus.innerHTML = `
            <div class="status-indicator ${isDetected ? 'active' : 'inactive'}"></div>
            Line Status: ${isDetected ? 'Detected' : 'Lost'}
        `;

        // Position with animated numbers
        this.animateNumber('position-x', data.position[0]);
        this.animateNumber('position-y', data.position[1]);
        this.animateNumber('position-theta', data.position[2]);

        // Command with glow effect
        const commandElement = document.getElementById('command');
        commandElement.textContent = `Command: ${data.last_command || 'None'}`;
        
        if (data.last_command) {
            commandElement.classList.add('command-active');
            setTimeout(() => commandElement.classList.remove('command-active'), 300);
        }

        // Objects detected
        document.getElementById('objects-count').textContent = data.detected_objects.length;
        
        // Update metrics
        this.metrics.objectsDetected = data.detected_objects.length;
        this.metrics.accuracy = isDetected ? 100 : 0;
    }

    updatePath(pathHistory) {
        if (!pathHistory || pathHistory.length === 0) return;

        this.pathHistory = pathHistory;
        this.drawPath();
    }

    drawPath() {
        const canvas = this.pathCanvas;
        const ctx = this.pathCtx;
        
        // Clear canvas with fade effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (this.pathHistory.length < 2) return;

        // Draw path with gradient
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#00ff88');
        gradient.addColorStop(1, '#0088ff');
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        ctx.beginPath();
        this.pathHistory.forEach((point, i) => {
            const x = (point[0] * 50) + canvas.width / 2;
            const y = (point[1] * 50) + canvas.height / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw current position with pulsing effect
        if (this.pathHistory.length > 0) {
            const lastPoint = this.pathHistory[this.pathHistory.length - 1];
            const x = (lastPoint[0] * 50) + canvas.width / 2;
            const y = (lastPoint[1] * 50) + canvas.height / 2;
            
            const time = Date.now() * 0.005;
            const radius = 8 + Math.sin(time) * 3;
            
            ctx.fillStyle = '#ff6b6b';
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    updateMetrics(data) {
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        
        // Update chart data
        if (this.metricsChart.data.labels.length > 20) {
            this.metricsChart.data.labels.shift();
            this.metricsChart.data.datasets[0].data.shift();
            this.metricsChart.data.datasets[1].data.shift();
        }
        
        this.metricsChart.data.labels.push(timeLabel);
        this.metricsChart.data.datasets[0].data.push(this.metrics.accuracy);
        this.metricsChart.data.datasets[1].data.push(this.metrics.speed);
        this.metricsChart.update('none');
    }

    animateNumber(elementId, targetValue) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const currentValue = parseFloat(element.textContent) || 0;
        const difference = targetValue - currentValue;
        const steps = 10;
        const stepValue = difference / steps;
        
        let step = 0;
        const animate = () => {
            if (step < steps) {
                const newValue = currentValue + (stepValue * step);
                element.textContent = newValue.toFixed(2);
                step++;
                requestAnimationFrame(animate);
            } else {
                element.textContent = targetValue.toFixed(2);
            }
        };
        
        animate();
    }

    handleKeyPress(key) {
        const commands = {
            'w': 'FORWARD',
            'a': 'LEFT',
            'd': 'RIGHT',
            's': 'STOP',
            'q': 'TURN_AROUND'
        };
        
        if (commands[key.toLowerCase()]) {
            this.socket.emit('manual_command', commands[key.toLowerCase()]);
            this.showNotification(`Manual command: ${commands[key.toLowerCase()]}`);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    startAnimations() {
        // Continuous path animation
        const animatePath = () => {
            this.drawPath();
            requestAnimationFrame(animatePath);
        };
        animatePath();

        // Update uptime
        setInterval(() => {
            this.metrics.uptime++;
            document.getElementById('uptime').textContent = this.formatUptime(this.metrics.uptime);
        }, 1000);
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new RobotDashboard();
}); 