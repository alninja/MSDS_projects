# Python Mini Projects - Assignment 2

This repository contains the complete implementation for **Assignment 2**, consisting of two mini-projects:

1. **Mini-Project 1**: Fetching and Displaying a Weather Forecast with Python Web APIs (`project1_weather.ipynb`)
2. **Mini-Project 2**: Uganda Tax Engine (`project2_taxes.ipynb`)

---

## Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── weather_client.py       # Reusable HTTP client, weather parsing, and retry wrappers
│   └── tax_engine.py           # Reusable Money value object, PAYE, VAT, CIT, and Payroll pipeline
├── tests/
│   ├── sample_geocoding.json   # Sample offline geocoding fixture
│   ├── sample_forecast.json    # Sample offline forecast fixture
│   └── test_tax_engine.py      # Boundary and Hypothesis property-based tests
├── project1_weather.ipynb      # Mini-Project 1 Notebook
├── project2_taxes.ipynb        # Mini-Project 2 Notebook
├── requirements.txt            # Project dependencies
└── README.md                   # Setup instructions and summary
```

---

## Setup & Installation

1. **Clone the repository**

```bash
# add the repository URL in place of <repo_url>
   git clone <repo_url>
   cd <repository-folder>
```

2. **Create and activate a virtual environment (recommended)**

```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
```

---

## Running the Notebooks

Both notebooks are fully reproducible and designed to run top-to-bottom using **Restart & Run All**:

- **Mini-Project 1:** Open `project1_weather.ipynb`. Supply your university contact email in the configuration cell (`USER_EMAIL`), then run all cells.
- **Mini-Project 2:** Open `project2_taxes.ipynb` and run all cells to execute the domain model, URA tax pipelines, and scenario analyses.

---

## Running Unit and Property-Based Tests

To run the test suite offline, including boundary checks and `hypothesis` property-based tests:

```bash
python -m pytest -v
```

---

## Summary of Findings and Implementation Highlights

- **Mini-Project 1:** Implemented HTTP communication using both `urllib` and `requests` for comparison. Integrated the Open-Meteo APIs with custom retry wrappers featuring exponential backoff, strict timeout management, and custom exception handling (`WeatherApiError`). Visualized multi-day temperature ranges against precipitation distributions using `matplotlib` and `pandas`.
- **Mini-Project 2:** Built an extensible, testable tax engine following strict OOP principles. Used `decimal.Decimal` exclusively, wrapped in a `Money` value object with operator overloading (`__add__`, `__sub__`, `__mul__`, `__lt__`) to avoid floating-point rounding errors. Modelled progressive monthly resident PAYE schedules, NSSF contributions, Local Service Tax, VAT (standard, zero-rated, exempt), and Corporate Income Tax (CIT).

---

## AI Declaration

In accordance with academic integrity guidelines, AI coding assistants were used during development of this repository to:

- Draft boilerplate structure for HTTP exception handling wrappers and exponential backoff logic.
- Format Markdown reports and documentation templates.

