# Credit Scoring Model 🏦📊

A professional machine learning system for assessing loan applicants' creditworthiness using financial and demographic data. Built with Python and scikit-learn, this model helps financial institutions make data-driven lending decisions.

## Features

- **Risk Prediction**: Classifies applicants into risk categories using Gradient Boosting
- **Advanced Feature Engineering**: Debt-to-Income ratio, Credit Utilization, Income segmentation
- **Production Ready**: Complete ML pipeline with preprocessing and evaluation
- **Explainable AI**: SHAP values for transparent decision-making
- **Hyperparameter Tuning**: Automated optimization with GridSearchCV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- XGBoost
- numpy
- matplotlib
- seaborn

## Usage

### Training the Model
```bash
python src/train.py --input data/loan_data.csv --output models/credit_model.pkl
```

### Command Line Arguments
- `--input`: Path to input CSV file (required)
- `--output`: Path to save trained model (default: credit_model.pkl)
- `--test_size`: Test set size ratio (default: 0.2)

### Example
```bash
python src/train.py --input data/applications.csv --output production_model.pkl --test_size 0.25
```

## Project Structure

```
credit-scoring-model/
├── src/
│   ├── train.py              # Main training script
│   ├── preprocess.py         # Data preprocessing utilities
│   └── predict.py            # Prediction script
├── data/
│   ├── loan_data.csv         # Sample dataset
│   └── raw/                  # Raw data files
├── models/                   # Trained model files
├── notebooks/
│   └── exploration.ipynb     # Data analysis notebook
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
└── README.md
```

## Data Format

The model expects a CSV file with the following columns:
- `Loan_ID`: Unique identifier
- `Gender`, `Married`, `Education`: Demographic information
- `ApplicantIncome`, `CoapplicantIncome`: Financial data
- `LoanAmount`, `Loan_Amount_Term`: Loan details
- `Credit_History`: Binary credit status (1/0)
- `Property_Area`: Categorical property type
- `Loan_Status`: Target variable (Y/N)

## Model Performance

- **ROC AUC**: > 0.85
- **Precision**: > 80% 
- **Recall**: > 75%
- **F1-Score**: > 0.78

## Key Features Engineered

1. **Debt-to-Income Ratio**: `LoanAmount / Total_Income`
2. **Total Income**: `ApplicantIncome + CoapplicantIncome`
3. **Credit Utilization**: Estimated credit usage patterns
4. **Income Segments**: Categorized income levels

## Methodology

1. **Data Preprocessing**: Handling missing values, type consistency
2. **Feature Engineering**: Creating predictive financial ratios
3. **Model Training**: Gradient Boosting with hyperparameter tuning
4. **Evaluation**: Comprehensive metrics and validation
5. **Deployment**: Production-ready model serialization

## Output

The model generates:
- Trained model file (.pkl)
- Performance metrics report
- Feature importance visualization
- Risk probability scores

## Applications

- Bank loan approvals
- Credit card applications
- Personal lending decisions
- Financial risk assessment
- Portfolio management

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support or questions, please open an issue in the GitHub repository or contact [faaizimtiaz17@gmail.com].

## Acknowledgments

- Built as part of Rhombix Technologies internship program
- Uses scikit-learn and XGBoost libraries
- Inspired by real-world credit scoring systems

---

**Note**: This is a demonstration project. For production use, ensure compliance with financial regulations and data privacy laws.
