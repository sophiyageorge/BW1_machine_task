# Simplified MLOps Pipeline for Machine Learning Project

## 📌 Project Overview
This project demonstrates a **simplified MLOps pipeline** for a machine learning model using **Python, FastAPI, Docker, GitHub Actions (CI/CD), and cloud deployment**.  
It includes automated testing, monitoring, logging, and containerization of the model for scalable deployment.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.10  
- **Machine Learning Library:** scikit-learn  
- **API Framework:** FastAPI  
- **Containerization:** Docker  
- **CI/CD:** GitHub Actions  
- **Monitoring & Metrics:** Prometheus & Grafana  
- **Logging:** Python Structured Logging  
- **Cloud Deployment:** AWS EC2 / Azure App Service / GCP Compute Engine  

---

## 📁 Project Structure

mlops-project/
│
├── app/
│ ├── train.py # Train ML model
│ ├── predict.py # FastAPI endpoints
│ └── model.pkl # Trained ML model
│
├── tests/
│ ├── test_model.py # Unit tests for model
│ ├── test_api.py # API endpoint tests
│ └── test_data.py # Data validation tests
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml # For Prometheus & Grafana
├── README.md
└── .github/workflows/ci-cd.yml


---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd mlops-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python app/train.py
```

### 4. Run API Locally
```bash
uvicorn app.predict:app --host 0.0.0.0 --port 8000
```
Visit: http://localhost:8000/docs for Swagger UI.

## 🐳 Docker Containerization

### 1. Build Docker Image
```bash
docker build -t mlops-project .
```

### 2. Run Container
```bash
docker run -p 8000:8000 mlops-project
```

### 3. Push to Docker Hub
```bash
docker tag mlops-project yourusername/mlops-project
docker push yourusername/mlops-project
```

## ☁️ Cloud Deployment

1. Launch an EC2 instance (Ubuntu) / Azure App Service / GCP VM.
2. Install Docker:
```bash
sudo apt update
sudo apt install docker.io -y
```
3. Pull and run Docker image:
```bash
docker pull yourusername/mlops-project
docker run -d -p 80:8000 yourusername/mlops-project
```
4. Access API:
```
http://<CLOUD_PUBLIC_IP>/docs
```

## 🧪 Automated Testing (Advanced)

### Test Coverage
- **Unit Tests:** Validate model outputs and data shapes.
- **API Tests:** Validate FastAPI endpoints.
- **Data Validation Tests:** Ensure correct input shape.

### Run Tests Locally
```bash
pytest --cov=app
```

## 🔄 CI/CD Pipeline

Automated using GitHub Actions
- Runs on every push to main branch
- **Steps:**
  - Checkout code
  - Setup Python environment
  - Install dependencies
  - Run tests
  - Build Docker image

## 📊 Logging & Monitoring

### 1. Structured Logging
Logs inputs, predictions, and errors using Python's logging module.

### 2. Prometheus Metrics
API tracks:
- Request count
- Request latency

```python
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 3. Grafana Dashboard
Connect Grafana → Prometheus → FastAPI /metrics

Visualize:
- API request rate
- Latency
- Prediction trends

### 4. Optional: ELK Stack
Can be integrated for centralized logging

## 🧩 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check / home |
| /predict | POST | Make prediction with input data |
| /metrics | GET | Prometheus metrics |

### Example Prediction Request

```http
POST /predict
[5.1, 3.5, 1.4, 0.2]
```

### Example Response

```json
{
  "prediction": [0]
}
```

## 📈 CI/CD Workflow

- **Trigger:** On push to main
- **Jobs:**
  - **Test:** Runs all unit, API, and data tests
  - **Docker Build:** Builds Docker image
  - **Deploy:** (Optional) Integrate with cloud deployment scripts

## 🔮 Future Improvements

- Model versioning with MLflow
- Data drift detection
- Scheduled retraining using Airflow
- Integration with real datasets (NLP, stock market, recommendations)
- Complete ELK Stack logging and alerting
- Auto-deployment to cloud using Terraform / Ansible

## 📌 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [scikit-learn](https://scikit-learn.org/)

## 📝 Author

**Mary Sophiya**

GitHub: https://github.com/sophiyageorge
LinkedIn: https://www.linkedin.com/in/marysophiya/