# Network Anomaly Detector

This project is a Python-based tool for detecting anomalies in network data. It leverages machine learning models and data preprocessing techniques to identify unusual patterns that may indicate network issues or security threats.

## Features
- Data preprocessing and cleaning
- Machine learning models for anomaly detection
- Configurable settings
- Sample data for testing

## Project Structure
- `app.py` - Main application entry point
- `config.py` - Configuration settings
- `models.py` - Machine learning models
- `preprocessing.py` - Data preprocessing utilities
- `utils.py` - Helper functions
- `sample.csv` - Example network data
- `requirements.txt` - Python dependencies

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd network-anomaly-detector
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your network data in CSV format (see `sample.csv` for an example).
2. Run the main application:
   ```bash
   streamlit run app.py 
   ```
3. Follow the prompts or configure settings in `config.py` as needed.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 