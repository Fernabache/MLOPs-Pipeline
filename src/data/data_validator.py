import os
from typing import Dict, List, Optional

import great_expectations as ge
import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

class DataValidator:
    def __init__(
        self,
        expectations_path: str = "configs/data_validation/expectations.json",
        data_path: Optional[str] = None
    ):
        """Initialize the data validator.
        
        Args:
            expectations_path: Path to the expectations configuration file
            data_path: Optional path to the data file
        """
        self.expectations_path = expectations_path
        self.data_path = data_path
        self.suite = self._load_expectations()
        
    def _load_expectations(self) -> ExpectationSuite:
        """Load expectations from configuration file."""
        if not os.path.exists(self.expectations_path):
            raise FileNotFoundError(f"Expectations file not found: {self.expectations_path}")
            
        context = ge.data_context.DataContext()
        return context.get_expectation_suite("default")
        
    def validate_dataset(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None
    ) -> Dict:
        """Validate a dataset against defined expectations.
        
        Args:
            data: Pandas DataFrame to validate
            data_path: Path to data file (if data not provided directly)
            
        Returns:
            Dictionary containing validation results
        """
        if data is None:
            if data_path is None:
                data_path = self.data_path
            if data_path is None:
                raise ValueError("Either data or data_path must be provided")
            data = pd.read_csv(data_path)
            
        ge_dataset = PandasDataset(data, expectation_suite=self.suite)
        results = ge_dataset.validate()
        
        return {
            "success": results.success,
            "statistics": results.statistics,
            "results": [
                {
                    "expectation": result.expectation_config.expectation_type,
                    "success": result.success,
                    "details": result.result
                }
                for result in results.results
            ]
        }
        
    def get_failed_validations(self, validation_results: Dict) -> List[Dict]:
        """Extract failed validations from results.
        
        Args:
            validation_results: Results from validate_dataset()
            
        Returns:
            List of failed validations with details
        """
        return [
            result for result in validation_results["results"]
            if not result["success"]
        ]
        
    def create_validation_report(
        self,
        validation_results: Dict,
        output_path: str = "validation_report.html"
    ) -> None:
        """Generate a human-readable validation report.
        
        Args:
            validation_results: Results from validate_dataset()
            output_path: Path to save the HTML report
        """
        failed = self.get_failed_validations(validation_results)
        
        html_content = f"""
        <html>
            <head>
                <title>Data Validation Report</title>
                <style>
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Data Validation Report</h1>
                <p>Overall Success: <span class="{
                    'success' if validation_results['success'] else 'failure'
                }">{validation_results['success']}</span></p>
                
                <h2>Failed Validations</h2>
                <ul>
                    {"".join([
                        f"<li>{failure['expectation']}: {failure['details']}</li>"
                        for failure in failed
                    ])}
                </ul>
                
                <h2>Statistics</h2>
                <pre>{validation_results['statistics']}</pre>
            </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)

if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Validate a dataset
    data = pd.read_csv("path/to/your/data.csv")
    results = validator.validate_dataset(data)
    
    # Generate report
    validator.create_validation_report(results)
    
    # Check for failures
    failed_validations = validator.get_failed_validations(results)
    if failed_validations:
        print("Validation failed!")
        for failure in failed_validations:
            print(f"- {failure['expectation']}: {failure['details']}")
    else:
        print("All validations passed!")
