# India-Canada FTA Export Impact Simulator

## Run
1. Install Python 3.11+.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

The app uses the bundled 2023 Canada tariff/import CSV and Canada HS6 import-demand elasticity workbook. It defaults to Armington elasticity 1.5, export-supply elasticity 99, and full tariff elimination.

## Key design controls
- Canada is fixed as importer/reporter; India (partner code 356) is the beneficiary exporter.
- MFN rows are used to prevent double counting duplicated AHS/MFN observations.
- Total export effect = trade creation + trade diversion + price effect. At export-supply elasticity 99, price effect is zero.
- Missing elasticities are flagged and excluded by default, with a visible median-imputation sensitivity option.
- Results can be filtered, ranked, charted and downloaded.

## Validation caveat
The model is a transparent deterministic prototype. Benchmark selected products against live WITS SMART outputs before policy use, especially the trade-diversion component.
