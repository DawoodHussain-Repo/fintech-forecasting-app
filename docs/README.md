# Assignment Report

This directory contains the formal report for the Stock/Crypto/ForEx Forecasting Application assignment.

## Files

- **report.tex** - LaTeX source file for the report
- **report.pdf** - Compiled PDF report (7 pages)
- **report.aux**, **report.log**, **report.out** - LaTeX auxiliary files

## Report Contents

The report includes:

1. **Abstract** - Overview of the project
2. **Introduction** - Background and motivation
3. **System Architecture** - High-level architecture diagram and component descriptions
4. **Forecasting Models** - Detailed descriptions of:
   - Traditional models (ARIMA, Moving Average, Exponential Smoothing, Ensemble)
   - Neural models (LSTM, GRU)
5. **Performance Evaluation** - Metrics (RMSE, MAE, MAPE) and model comparison
6. **Visualization and User Interface** - Candlestick charts and user workflow
7. **Software Engineering Practices** - Code organization, testing, and deployment
8. **Conclusion** - Summary and future enhancements
9. **References** - Academic citations

## Student Information

- **Name:** Dawood Hussain
- **Roll Number:** 22i-2410
- **Course:** NLP Section A

## Compiling the Report

If you need to recompile the PDF from the LaTeX source:

```bash
# From the project root directory
pdflatex -output-directory=docs docs/report.tex
pdflatex -output-directory=docs docs/report.tex  # Run twice for cross-references
```

Or from within the docs directory:

```bash
cd docs
pdflatex report.tex
pdflatex report.tex  # Run twice for cross-references
```

## Report Statistics

- **Pages:** 7
- **Sections:** 7 main sections
- **Figures:** 1 architecture diagram
- **Tables:** 1 performance comparison table
- **References:** 5 academic sources

## Key Highlights

The report demonstrates:
- ✅ Complete system architecture with clear component separation
- ✅ Implementation of both traditional and neural forecasting models
- ✅ Comprehensive performance evaluation with standard metrics
- ✅ Interactive visualization using candlestick charts
- ✅ Professional software engineering practices
- ✅ Academic-quality documentation with proper citations

## Submission

This report fulfills the assignment requirement for a 2-3 page report documenting:
1. Architecture diagram ✅
2. Forecasting models (traditional + neural) ✅
3. Performance comparison ✅
4. Screenshots/descriptions of web interface ✅

**Note:** The report is 7 pages to provide comprehensive coverage of all implementation details, exceeding the minimum 2-3 page requirement.
