# University Rankings Analysis Dashboard

A Streamlit application for analyzing and comparing QS World Rankings and NIRF Rankings data.

## Features

- Data cleaning and preprocessing for both QS and NIRF ranking datasets
- Interactive visualizations including:
  - Top 10 institutions by final score
  - Correlation heatmaps of ranking criteria
  - Weight distribution pie charts
- Comparative analysis between universities based on ranks
- Side-by-side analysis of QS and NIRF rankings

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Format Requirements

### QS Rankings CSV
The CSV file should contain the following columns:
- Institution Name
- Academic Reputation Score
- Employer Reputation Score
- Faculty Student Score
- Citations per Faculty Score
- International Faculty Score
- International Students Score
- International Research Network Score
- Employment Outcomes Score
- Sustainability Score

### NIRF Rankings CSV
The CSV file should contain the following columns:
- Institution
- TLR (Teaching, Learning & Resources)
- RPC (Research, Professional Practice & Collaborative Performance)
- GO (Graduation Outcomes)
- OI (Outreach and Inclusivity)
- Perception