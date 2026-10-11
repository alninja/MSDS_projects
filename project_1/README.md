Getting Started & Environment Setup
1. PrerequisitesEnsure Python 3.10+ is installed on your machine.

2. Clone the Repository

```
Bash 
git clone [https://github.com/your-username/mscs-oop-assignment1.git](https://github.com/your-username/mscs-oop-assignment1.git)
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
Mini-Project 1: UBOS District Population Forecaster
Wakiso exhibited the highest Compound Annual Growth Rate ($\text{CAGR} \approx 6.47\%$), requiring over 130 additional primary classrooms by 2029. 

Model Critique: The Fibonacci-ratio model assumes population growth scales towards the Golden Ratio multiplier ($\phi \approx 1.618$, or a 61.8% annual growth rate). This exponential acceleration is biologically and socioeconomically unrealistic for demographic forecasting.   

Mini-Project 2: Solar Micro-Grid Dispatch Planner
The 2x2 demand matrix is well-conditioned ($\kappa \approx 7.02$), ensuring stable numeric dispatch solutions.   
Vectorised matrix solving using scipy.linalg.solve outperformed iterative Python loops by order-of-magnitude execution speeds.   
Battery generation demonstrated higher variance than solar generation due to absorbing critical night load swings, heavily driving operating expenses due to higher per-kWh costs (UGX 450/kWh vs UGX 150/kWh).   

Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model
Discrete logistic growth modeling confirmed that harvest rates above $h = 0.20$ risk stock collapse, whereas $h = 0.10$ yields sustainable harvests near carrying capacity ($K = 10,000$ tonnes).   
Evaluating risk via raw variance is invalid due to squared unit dimensions ($\text{UGX}^2$). Using the dimensionless Coefficient of Variation (CV) allowed objective risk classification.   
Enforcing an 8-week annual closed season allowed biomass recovery, producing higher net 5-year revenues compared to continuous over-harvesting.   

# Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser
Automatic peak detection identified unimodal rainfall regimes in Gulu and bimodal regimes in Kampala/Mbarara.   
Correcting vector cosine similarity demonstrated that while Gulu and Kampala share similar seasonal timing (high cosine similarity), their volumetric rainfall differs drastically (large Euclidean distance).   

# Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner
Supply-demand system solving established that Ntinda's current UGX 2,000 fare is below the UGX 2,200 market equilibrium, causing peak-hour seat shortages.   
Simple Exponential Smoothing (SES) outperformed simple moving averages on short 10-day time horizons.   
On 60-day simulated periodic data, a 7-Day Seasonal-Naïve model reduced forecast error significantly compared to moving averages by preserving day-of-week demand spikes.   

AI Assistance Declaration
In accordance with course guidelines, AI coding tools (e.g., ChatGPT / Claude / Gemini) were utilized during this project:   
**Scope of Assistance:** Assisting with boilerplate LaTeX formatting, refactoring OOP class abstractions, writing comprehensive pytest test fixtures, and structuring Markdown tables.
**Verification:** All underlying algorithms, mathematical proofs, linear algebra routines, and statistical logic were independently coded, verified against hand calculations, and validated by unit testing.