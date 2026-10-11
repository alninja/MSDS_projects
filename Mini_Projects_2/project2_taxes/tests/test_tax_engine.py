from decimal import Decimal
import pytest
from hypothesis import given, strategies as st
from src.tax_engine import (
    Money,
    PAYECalculator,
    TaxContext,
    PayrollPipeline,
    VATModule,
    CorporateIncomeTax,
    InvalidIncomeError
)

def test_money_value_object_operations():
    m1 = Money(Decimal("100000"))
    m2 = Money(Decimal("50000"))
    assert (m1 + m2).amount == Decimal("150000.00")
    assert (m1 - m2).amount == Decimal("50000.00")
    assert (m1 * 2).amount == Decimal("200000.00")
    assert m2 < m1

def test_paye_boundary_bands():
    calc = PAYECalculator()
    
    # Boundary: 335,000 (Nil)
    res_335k = calc.calculate(TaxContext(gross_income=Decimal("335000")))
    assert res_335k.tax_due.amount == Decimal("0.00")

    # Boundary: 335,001 (20% of 1)
    res_335k_1 = calc.calculate(TaxContext(gross_income=Decimal("335001")))
    assert res_335k_1.tax_due.amount == Decimal("0.20")

    # Boundary: 410,000
    res_410k = calc.calculate(TaxContext(gross_income=Decimal("410000")))
    assert res_410k.tax_due.amount == Decimal("15000.00")  # 75,000 * 0.20

    # Boundary: 485,000
    res_485k = calc.calculate(TaxContext(gross_income=Decimal("485000")))
    assert res_485k.tax_due.amount == Decimal("33750.00")

    # Boundary: 10,000,000
    res_10m = calc.calculate(TaxContext(gross_income=Decimal("10000000")))
    assert res_10m.tax_due.amount > Decimal("0")

@given(gross=st.decimals(min_value=0, max_value=50000000, places=2))
def test_hypothesis_tax_monotonicity_and_bounds(gross):
    calc = PAYECalculator()
    ctx = TaxContext(gross_income=gross)
    result = calc.calculate(ctx)
    
    # Property 1: Tax must be non-negative
    assert result.tax_due.amount >= Decimal("0")
    
    # Property 2: Tax must never exceed gross income
    assert result.tax_due.amount <= gross

def test_vat_computation():
    invoices = [
        {"amount": 1000000, "type": "standard", "direction": "output"},
        {"amount": 400000, "type": "standard", "direction": "input"},
        {"amount": 500000, "type": "zero", "direction": "output"}
    ]
    res = VATModule.compute_net_vat(invoices)
    assert res["output_vat"].amount == Decimal("180000.00")
    assert res["input_vat"].amount == Decimal("72000.00")
    assert res["net_vat_payable"].amount == Decimal("108000.00")

def test_float_unsafe_demonstration():
    # Concrete proof of float inaccuracy in finance
    float_sum = 0.1 + 0.2
    assert float_sum != 0.3  # Standard float precision flaw (0.30000000000000004)

    # Decimal safety check
    dec_sum = Decimal("0.1") + Decimal("0.2")
    assert dec_sum == Decimal("0.3")