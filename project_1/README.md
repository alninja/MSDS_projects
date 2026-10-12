Getting Started & Environment Setup
1. Prerequisites
Ensure Python 3.10+ is installed on your machine.

2. Clone the Repository

```
Bash 
git clone https://github.com/alninja/MSDS_projects/tree/b96cdce2d2b1141e6d87ecce7108c2c2520aca4a/project_1 (https://github.com/your-username/mscs-oop-assignment1.git)
cd mscs-oop-assignment1
```

3. Create and Activate a Virtual Environment

Linux/macOS:
```
Bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt):
```
DOS
python -m venv venv
venv\Scripts\activate
```

4. Install Dependencies

```
Bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Running Unit Tests

Unit tests are written using pytest to cover normal cases, edge cases (zero/negative values, empty inputs), and matrix validations across all 5 projects. To execute the test suite:   
```
Bash
pytest
```

To run tests with detailed verbosity:
```
Bash
pytest -v
```

# Running the Notebooks
Launch Jupyter Notebook or JupyterLab:
```
Bash
jupyter notebook
```

Navigate to the notebooks/ directory and open any project notebook (e.g., project1_population.ipynb). All notebooks use fixed random seeds (np.random.default_rng(seed)) and are designed to run top-to-bottom without errors via Kernel -> Restart & Run All.   

Summary of Findings & Key Results
# Mini-Project 1: UBOS District Population Forecaster

- **Growth Leader:** Wakiso district recorded the highest annual growth rate ($\text{CAGR} \approx 6.47\%$), meaning it will need over 130 new primary classrooms by 2029.
- **Model Critique:** The Fibonacci-ratio model assumes an explosive annual growth rate of 61.8% ($\phi \approx 1.618$). This is unrealistic for long-term population planning because real human growth is bounded by resource limits and carrying capacity.

---

# Mini-Project 2: Solar Micro-Grid Dispatch Planner

- **System Stability:** The micro-grid demand matrix is well-conditioned ($\kappa \approx 7.02$), so small changes in daily load do not cause large errors in energy calculations.
- **Performance:** Vectorized matrix solving (`scipy.linalg.solve`) was significantly faster than iterative Python loops.
- **Cost Drivers:** Battery power had much higher variance than solar because it absorbs critical night load swings. Batteries cost three times more per kWh than solar (UGX 450 vs. UGX 150), so peak night demand heavily drives up operational expenses.

---

# Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

- **Harvest Sustainability:** Discrete logistic modelling showed that keeping the harvest rate near $h = 0.10$ maintains stable fish stocks close to carrying capacity (10,000 tonnes), whereas rates above $h = 0.20$ risk population collapse.
- **Risk Metrics:** Raw variance is not useful for measuring revenue risk because its units are squared ($\text{UGX}^2$). The Coefficient of Variation (CV) provided a reliable, scale-free measure of risk.
- **Closed Seasons:** Enforcing an 8-week annual fishing ban allows the fish population to recover, generating higher cumulative revenue over a 5-year period than continuous over-harvesting.

---

# Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser

- **Climate Regimes:** Automatic peak detection correctly identified a unimodal rainfall pattern in Gulu and bimodal patterns in Kampala and Mbarara.
- **Similarity Pitfall:** Comparing regions using cosine similarity alone can be misleading. Gulu and Kampala have similar seasonal rainfall timing (high cosine similarity), but their actual water volumes differ drastically (large Euclidean distance). Extension officers must look at total volume, not just timing, when advising farmers.

---

# Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

- **Equilibrium Pricing:** Solving the supply-demand system for the Ntinda route showed that the current UGX 2,000 fare is below the market equilibrium of UGX 2,200, which causes rush-hour passenger seat shortages.
- **Forecasting:** Simple Exponential Smoothing (SES) and Linear Trend models outperformed 3-day moving averages on short 10-day histories.
- **Seasonal Demand:** Over a 60-day simulation with weekly patterns (busier Fridays, quieter Sundays), a 7-day Seasonal-Naïve forecaster beat standard moving averages by capturing weekend and weekday demand swings.   

AI Assistance Declaration
In accordance with course guidelines, AI coding tools (e.g., ChatGPT / Claude / Gemini) were utilized during this project:   
**Scope of Assistance:** Assisting with boilerplate LaTeX formatting, refactoring OOP class abstractions, writing comprehensive pytest test fixtures, and structuring Markdown tables.
**Verification:** All underlying algorithms, mathematical proofs, linear algebra routines, and statistical logic were independently coded, verified against hand calculations, and validated by unit testing.