from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Union

# --- Custom Exception Hierarchy ---
class TaxError(Exception):
    """Base exception for all tax-related errors."""
    pass

class InvalidIncomeError(TaxError):
    """Raised when income or monetary inputs are invalid (negative or non-numeric)."""
    pass

class NoScheduleForDateError(TaxError):
    """Raised when no tax schedule matches the given parameters or date."""
    pass

# --- Money Value Object ---
@dataclass(frozen=True)
class Money:
    amount: Decimal

    def __post_init__(self):
        try:
            dec_val = Decimal(str(self.amount))
        except Exception as e:
            raise InvalidIncomeError(f"Invalid monetary amount: {self.amount}") from e
        
        if dec_val < Decimal('0'):
            raise InvalidIncomeError(f"Monetary amount cannot be negative: {dec_val}")
            
        object.__setattr__(self, 'amount', dec_val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def __add__(self, other: 'Money') -> 'Money':
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount + other.amount)

    def __sub__(self, other: 'Money') -> 'Money':
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount - other.amount)

    def __mul__(self, factor: Union[int, float, Decimal]) -> 'Money':
        return Money(self.amount * Decimal(str(factor)))

    def __lt__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount < other.amount

    def __le__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount <= other.amount

    def __gt__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount > other.amount

    def __ge__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount >= other.amount

    def __repr__(self) -> str:
        return f"UGX {self.amount:,.2f}"

# --- Context and Result Classes ---
@dataclass
class TaxContext:
    gross_income: Decimal
    allowances: Decimal = Decimal('0')
    is_resident: bool = True
    additional_params: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class TaxResult:
    tax_due: Money
    effective_rate: Decimal
    breakdown: Dict[str, Any]

# --- Abstract Base Class ---
class Tax(ABC):
    @abstractmethod
    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Compute tax liability based on the context."""
        pass

# --- PAYE Calculator Module ---
class PAYECalculator(Tax):
    """
    Uganda Revenue Authority (URA) Monthly Resident & Non-Resident PAYE Calculator.
    Baseline reference: URA tax schedules (Income Tax Act).
    """
    def calculate(self, ctx: TaxContext) -> TaxResult:
        taxable_income = Decimal(str(ctx.gross_income)) - Decimal(str(ctx.allowances))
        if taxable_income < Decimal('0'):
            raise InvalidIncomeError("Taxable income after allowances cannot be negative.")

        tax_due = Decimal('0')
        breakdown_details = {}

        if ctx.is_resident:
            # Resident Monthly Bands
            # Band 1: Up to 335,000 -> Nil
            # Band 2: 335,001 to 410,000 -> 20% of excess over 335,000
            # Band 3: 410,001 to 485,000 -> 15,000 + 25% of excess over 410,000
            # Band 4: 485,001 to 10,000,000 -> 33,750 + 30% of excess over 485,000
            # Surcharge: Above 10,000,000 -> Add 10% of excess over 10,000,000
            
            remaining = taxable_income
            
            if taxable_income <= Decimal('335000'):
                tax_due = Decimal('0')
                breakdown_details["band"] = "Up to 335,000"
            else:
                # Accumulate progressive tax
                t = Decimal('0')
                
                # Band 2 portion (335,000 to 410,000)
                if taxable_income > Decimal('335000'):
                    b2_amt = min(taxable_income, Decimal('410000')) - Decimal('335000')
                    t += b2_amt * Decimal('0.20')
                
                # Band 3 portion (410,000 to 485,000)
                if taxable_income > Decimal('410000'):
                    b3_amt = min(taxable_income, Decimal('485000')) - Decimal('410000')
                    t += b3_amt * Decimal('0.25')
                
                # Band 4 portion (485,000 to 10,000,000)
                if taxable_income > Decimal('485000'):
                    b4_amt = min(taxable_income, Decimal('10000000')) - Decimal('485000')
                    t += b4_amt * Decimal('0.30')
                
                # Surcharge above 10,000,000 (10% surcharge)
                if taxable_income > Decimal('10000000'):
                    surcharge_amt = taxable_income - Decimal('10000000')
                    t += surcharge_amt * Decimal('0.10')
                    
                tax_due = t
        else:
            # Non-resident flat or specific schedule rate (e.g., 30% flat on employment income for non-residents)
            tax_due = taxable_income * Decimal('0.30')

        tax_money = Money(tax_due)
        effective_rate = (tax_due / taxable_income) if taxable_income > Decimal('0') else Decimal('0')

        return TaxResult(
            tax_due=tax_money,
            effective_rate=effective_rate.quantize(Decimal('0.0001')),
            breakdown={"taxable_income": taxable_income, "schedule": "Resident Monthly" if ctx.is_resident else "Non-Resident"}
        )

# --- Payroll Pipeline ---
class PayrollPipeline:
    def __init__(self, lst_amount: Decimal = Decimal('0')):
        self.lst_amount = Money(lst_amount)
        self.paye_calculator = PAYECalculator()

    def process_payroll(self, gross_pay: Decimal, allowances: Decimal = Decimal('0'), is_resident: bool = True) -> Dict[str, Any]:
        gross = Money(gross_pay)
        allow = Money(allowances)
        total_gross = gross + allow

        # NSSF Computations: Employee 5%, Employer 10%
        emp_nssf = gross * Decimal('0.05')
        er_nssf = gross * Decimal('0.10')

        # Chargeable income for PAYE (Gross minus Employee NSSF and LST)
        taxable_base = total_gross.amount - emp_nssf.amount - self.lst_amount.amount
        if taxable_base < Decimal('0'):
            taxable_base = Decimal('0')

        paye_ctx = TaxContext(gross_income=taxable_base, is_resident=is_resident)
        paye_result = self.paye_calculator.calculate(paye_ctx)
        paye_due = paye_result.tax_due

        # Net Pay = Total Gross - Employee NSSF - LST - PAYE
        net_pay = total_gross - emp_nssf - self.lst_amount - paye_due

        return {
            "gross_pay": gross,
            "allowances": allow,
            "total_gross": total_gross,
            "employee_nssf": emp_nssf,
            "employer_nssf": er_nssf,
            "local_service_tax": self.lst_amount,
            "paye": paye_due,
            "net_pay": net_pay
        }

# --- VAT Module ---
class VATModule:
    STANDARD_RATE = Decimal('0.18')

    @staticmethod
    def compute_net_vat(invoices: List[Dict[str, Any]]) -> Dict[str, Money]:
        output_vat = Decimal('0')
        input_vat = Decimal('0')

        for inv in invoices:
            amount = Decimal(str(inv.get('amount', '0')))
            vat_type = inv.get('type', 'standard')  # 'standard', 'zero', 'exempt'
            direction = inv.get('direction', 'output')  # 'output' (sales), 'input' (purchases)

            if vat_type == 'standard':
                vat_val = amount * VATModule.STANDARD_RATE
                if direction == 'output':
                    output_vat += vat_val
                else:
                    input_vat += vat_val
            elif vat_type in ['zero', 'exempt']:
                pass  # Zero-rated/exempt generate no VAT liability

        net_payable = output_vat - input_vat
        return {
            "output_vat": Money(output_vat),
            "input_vat": Money(input_vat),
            "net_vat_payable": Money(net_payable if net_payable > Decimal('0') else Decimal('0'))
        }

# --- Corporate Income Tax (CIT) Module ---
class CorporateIncomeTax(Tax):
    CIT_RATE = Decimal('0.30')  # 30% for resident companies

    def calculate(self, ctx: TaxContext) -> TaxResult:
        revenue = Decimal(str(ctx.gross_income))
        allowable_deductions = Decimal(str(ctx.allowances))
        chargeable_income = revenue - allowable_deductions
        
        if chargeable_income < Decimal('0'):
            chargeable_income = Decimal('0')

        tax_due = chargeable_income * self.CIT_RATE
        return TaxResult(
            tax_due=Money(tax_due),
            effective_rate=self.CIT_RATE,
            breakdown={"revenue": revenue, "deductions": allowable_deductions, "chargeable_income": chargeable_income}
        )

# --- Extension Task B: TaxPlanner (Salary vs Dividend Scenario Comparison) ---
class TaxPlanner:
    @staticmethod
    def compare_salary_vs_dividend(annual_gross_salary: Decimal, annual_dividend: Decimal) -> Dict[str, Any]:
        # Salary taxed via PAYE brackets (approximated monthly average or annualized)
        paye_calc = PAYECalculator()
        monthly_salary = annual_gross_salary / Decimal('12')
        monthly_paye_res = paye_calc.calculate(TaxContext(gross_income=monthly_salary))
        total_annual_paye = monthly_paye_res.tax_due * Decimal('12')

        # Dividend withholding tax in Uganda is typically 10% (resident)
        dividend_wht = Money(annual_dividend * Decimal('0.10'))

        return {
            "salary_scenario_tax": total_annual_paye,
            "dividend_scenario_tax": dividend_wht,
            "difference": total_annual_paye - dividend_wht
        }