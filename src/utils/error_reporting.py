#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced error reporting module for all workflows.
Creates detailed, stage-specific error reports across workflow runs.
"""
import os
import sys
import json
import re
import logging
import datetime
import traceback
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Set, Tuple

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the logging system
from engine.logging_system import LoggingSystem

# Initialize the logging system
logging_system = LoggingSystem(
    console_output=True,
    file_output=True,
    default_level='INFO',
    async_logging=True
)
logger = logging_system.get_logger('error_reporting')

# Define workflow stages for each workflow type
WORKFLOW_STAGES = {
    "simple": [
        "initialization", 
        "data_loading", 
        "strategy_setup", 
        "backtest_execution", 
        "performance_evaluation"
    ],
    "optimization": [
        "initialization", 
        "data_loading", 
        "parameter_grid_generation", 
        "optimization_iterations", 
        "best_parameter_selection", 
        "backtest_with_best_params", 
        "performance_evaluation"
    ],
    "monte_carlo": [
        "initialization", 
        "data_loading", 
        "strategy_setup", 
        "initial_backtest", 
        "simulation_generation", 
        "monte_carlo_execution", 
        "statistical_analysis", 
        "performance_evaluation"
    ],
    "walkforward": [
        "initialization", 
        "data_loading", 
        "window_generation", 
        "in_sample_optimization", 
        "out_of_sample_testing", 
        "combined_results_analysis", 
        "performance_evaluation"
    ],
    "complete": [
        "initialization", 
        "data_loading", 
        "simple_backtest", 
        "parameter_optimization", 
        "walkforward_analysis", 
        "monte_carlo_simulation", 
        "optimized_backtest", 
        "performance_evaluation"
    ]
}

# Error severity levels
ERROR_SEVERITY = {
    "CRITICAL": 3,  # System errors, exceptions, crashes
    "ERROR": 2,     # Process errors, logic errors
    "WARNING": 1,   # Potential issues, suspect results
    "INFO": 0       # Informational messages that indicate issues
}

class StageError:
    """
    Class to represent a workflow stage-specific error with detailed context.
    """
    def __init__(self, 
                 message: str, 
                 stage: str, 
                 severity: str = "ERROR",
                 exception: Optional[Exception] = None,
                 timestamp: Optional[datetime.datetime] = None,
                 file_path: Optional[str] = None,
                 line_number: Optional[int] = None,
                 context_before: Optional[List[str]] = None,
                 context_after: Optional[List[str]] = None):
        """
        Initialize a stage error.
        
        Args:
            message: Error message
            stage: Workflow stage where the error occurred
            severity: Error severity (CRITICAL, ERROR, WARNING, INFO)
            exception: Original exception object, if applicable
            timestamp: When the error occurred
            file_path: Source file where the error occurred
            line_number: Line number in source file
            context_before: Context lines before the error
            context_after: Context lines after the error
        """
        self.message = message
        self.stage = stage
        self.severity = severity.upper()  # Normalize to uppercase
        self.exception = exception
        self.timestamp = timestamp or datetime.datetime.now()
        self.file_path = file_path
        self.line_number = line_number
        self.context_before = context_before or []
        self.context_after = context_after or []
        
        # Extract traceback information if exception is provided
        if exception and not file_path:
            try:
                tb = traceback.extract_tb(exception.__traceback__)
                if tb:
                    # Get the last frame from the traceback for most specific info
                    frame = tb[-1]
                    self.file_path = frame.filename
                    self.line_number = frame.lineno
            except:
                pass  # Ignore traceback extraction errors
    
    @property 
    def severity_level(self) -> int:
        """Get the numeric severity level for sorting."""
        return ERROR_SEVERITY.get(self.severity, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "stage": self.stage,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "line_number": self.line_number,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "exception_type": type(self.exception).__name__ if self.exception else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageError':
        """Create a StageError from a dictionary."""
        timestamp = None
        if "timestamp" in data:
            try:
                timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            except:
                timestamp = datetime.datetime.now()
        
        return cls(
            message=data["message"],
            stage=data["stage"],
            severity=data.get("severity", "ERROR"),
            timestamp=timestamp,
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            context_before=data.get("context_before", []),
            context_after=data.get("context_after", [])
        )
    
    def __str__(self) -> str:
        """String representation of the error."""
        result = f"[{self.severity}] {self.stage}: {self.message}"
        if self.file_path:
            result += f" (in {os.path.basename(self.file_path)}"
            if self.line_number:
                result += f":{self.line_number}"
            result += ")"
        return result

class StageErrorReport:
    """
    Class to collect and manage stage-specific errors for a workflow run.
    """
    def __init__(self, 
                 workflow_type: str, 
                 strategy_name: str, 
                 output_dir: str):
        """
        Initialize a stage error report.
        
        Args:
            workflow_type: Type of workflow (simple, optimization, etc.)
            strategy_name: Name of the strategy
            output_dir: Directory where workflow results are stored
        """
        self.workflow_type = workflow_type
        self.strategy_name = strategy_name
        self.output_dir = output_dir
        self.errors_by_stage = defaultdict(list)
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        
        # Get expected stages for this workflow type
        self.expected_stages = WORKFLOW_STAGES.get(workflow_type, [])
        
    def add_error(self, error: StageError) -> None:
        """Add an error to the report."""
        self.errors_by_stage[error.stage].append(error)
        self.updated_at = datetime.datetime.now()
        
    def add_exception(self, 
                     exception: Exception, 
                     stage: str, 
                     message: Optional[str] = None,
                     severity: str = "CRITICAL") -> None:
        """
        Add an exception as an error.
        
        Args:
            exception: The exception object
            stage: Workflow stage where the exception occurred
            message: Optional custom message (defaults to str(exception))
            severity: Error severity level
        """
        error = StageError(
            message=message or str(exception),
            stage=stage,
            severity=severity,
            exception=exception
        )
        self.add_error(error)
        
    def get_all_errors(self) -> List[StageError]:
        """Get all errors across all stages."""
        all_errors = []
        for stage_errors in self.errors_by_stage.values():
            all_errors.extend(stage_errors)
        return all_errors
    
    def has_errors(self) -> bool:
        """Check if the report contains any errors."""
        return bool(self.get_all_errors())
    
    def has_critical_errors(self) -> bool:
        """Check if the report contains any critical errors."""
        for error in self.get_all_errors():
            if error.severity == "CRITICAL":
                return True
        return False
    
    def has_errors_in_stage(self, stage: str) -> bool:
        """Check if a specific stage has errors."""
        return stage in self.errors_by_stage and bool(self.errors_by_stage[stage])
    
    def has_critical_errors_in_stage(self, stage: str) -> bool:
        """Check if a specific stage has critical errors."""
        if stage not in self.errors_by_stage:
            return False
        
        for error in self.errors_by_stage[stage]:
            if error.severity == "CRITICAL":
                return True
        return False
    
    def get_stage_errors(self, stage: str) -> List[StageError]:
        """Get all errors for a specific stage."""
        return self.errors_by_stage.get(stage, [])
    
    def count_errors_by_severity(self) -> Dict[str, int]:
        """Count errors by severity level."""
        counts = defaultdict(int)
        for error in self.get_all_errors():
            counts[error.severity] += 1
        return dict(counts)
    
    def count_errors_by_stage(self) -> Dict[str, int]:
        """Count errors by stage."""
        return {stage: len(errors) for stage, errors in self.errors_by_stage.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary for serialization."""
        return {
            "workflow_type": self.workflow_type,
            "strategy_name": self.strategy_name,
            "output_dir": self.output_dir,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "errors_by_stage": {
                stage: [error.to_dict() for error in errors] 
                for stage, errors in self.errors_by_stage.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageErrorReport':
        """Create a report from a dictionary."""
        report = cls(
            workflow_type=data["workflow_type"],
            strategy_name=data["strategy_name"],
            output_dir=data["output_dir"]
        )
        
        # Restore timestamps
        try:
            report.created_at = datetime.datetime.fromisoformat(data["created_at"])
            report.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        except:
            pass  # Use default timestamps if parsing fails
        
        # Restore errors by stage
        for stage, error_dicts in data.get("errors_by_stage", {}).items():
            for error_dict in error_dicts:
                error = StageError.from_dict(error_dict)
                report.errors_by_stage[stage].append(error)
        
        return report
    
    def save(self, file_path: Optional[str] = None) -> str:
        """
        Save the report to a JSON file.
        
        Args:
            file_path: Path to save the report file. If None, a default path is used.
            
        Returns:
            Path to the saved file
        """
        if not file_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.strategy_name}_{self.workflow_type}_stage_errors_{timestamp}.json"
            file_path = os.path.join(self.output_dir, file_name)
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
                
            logger.info(f"Saved stage error report to: {file_path}")
            return file_path
        except PermissionError:
            # If we can't write to the original path, try an alternative location
            alt_dir = os.path.join(project_root, "logs")
            os.makedirs(alt_dir, exist_ok=True)
            
            alt_path = os.path.join(alt_dir, os.path.basename(file_path))
            try:
                with open(alt_path, 'w') as f:
                    json.dump(self.to_dict(), f, indent=2)
                logger.warning(f"Permission denied for original path. Saved report to: {alt_path}")
                return alt_path
            except Exception as e:
                logger.error(f"Failed to save report to alternative location: {e}")
                return "Error: Could not save report file"
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return "Error: Could not save report file"
    
    @classmethod
    def load(cls, file_path: str) -> 'StageErrorReport':
        """
        Load a report from a JSON file.
        
        Args:
            file_path: Path to the report file
            
        Returns:
            Loaded StageErrorReport object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def generate_report_text(self) -> str:
        """
        Generate a formatted text report.
        
        Returns:
            Text report content
        """
        lines = []
        
        # Report header
        lines.append("=" * 80)
        lines.append(f"{self.workflow_type.upper()} WORKFLOW STAGE ERROR REPORT".center(80))
        lines.append("=" * 80)
        
        # Basic information
        lines.append(f"Strategy: {self.strategy_name}")
        lines.append(f"Generated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Error statistics
        all_errors = self.get_all_errors()
        lines.append(f"Total errors: {len(all_errors)}")
        
        # Error counts by severity
        severity_counts = self.count_errors_by_severity()
        if severity_counts:
            lines.append("\nErrors by severity:")
            for severity, count in sorted(severity_counts.items(), 
                                         key=lambda x: ERROR_SEVERITY.get(x[0], 0), 
                                         reverse=True):
                lines.append(f"  {severity}: {count}")
        
        # Error counts by stage
        stage_counts = self.count_errors_by_stage()
        if stage_counts:
            lines.append("\nErrors by stage:")
            
            # First show the expected stages in workflow order
            for stage in self.expected_stages:
                if stage in stage_counts:
                    stage_display = stage.replace('_', ' ').title()
                    lines.append(f"  {stage_display}: {stage_counts[stage]}")
            
            # Then show any unexpected stages
            for stage, count in stage_counts.items():
                if stage not in self.expected_stages:
                    stage_display = stage.replace('_', ' ').title()
                    lines.append(f"  {stage_display}: {count}")
        
        # Show errors by stage
        stages_to_report = []
        
        # First add expected stages in the correct order
        for stage in self.expected_stages:
            if stage in self.errors_by_stage:
                stages_to_report.append(stage)
        
        # Then add any unexpected stages
        for stage in self.errors_by_stage:
            if stage not in self.expected_stages:
                stages_to_report.append(stage)
        
        # Generate detailed error report for each stage
        for stage in stages_to_report:
            stage_display = stage.replace('_', ' ').title()
            lines.append("\n" + "-" * 80)
            lines.append(f"Stage: {stage_display}")
            lines.append("-" * 80)
            
            # Sort errors by severity (critical first)
            stage_errors = sorted(self.errors_by_stage[stage], 
                                key=lambda e: (e.severity_level, e.timestamp), 
                                reverse=True)
            
            # Add each error
            for i, error in enumerate(stage_errors, 1):
                lines.append(f"\nError {i}: [{error.severity}] {error.message}")
                
                if error.file_path:
                    lines.append(f"Location: {error.file_path}" + 
                               (f":{error.line_number}" if error.line_number else ""))
                
                if error.context_before:
                    lines.append("\nContext before:")
                    for line in error.context_before:
                        lines.append(f"  {line}")
                
                if error.context_after:
                    lines.append("\nContext after:")
                    for line in error.context_after:
                        lines.append(f"  {line}")
                
                lines.append("-" * 40)  # Separator between errors
        
        # Final section: recommendations based on errors
        if all_errors:
            lines.append("\n" + "=" * 80)
            lines.append("RECOMMENDATIONS")
            lines.append("=" * 80)
            
            if self.has_critical_errors():
                lines.append("\nCritical errors were detected. Consider the following actions:")
                
                # Specific recommendations based on stages with critical errors
                for stage in self.expected_stages:
                    if self.has_critical_errors_in_stage(stage):
                        if stage == "initialization":
                            lines.append("- Check configuration settings and ensure all paths are correct")
                            lines.append("- Verify that all required modules are installed")
                        elif stage == "data_loading":
                            lines.append("- Ensure data files exist and are properly formatted")
                            lines.append("- Check that date ranges are valid")
                            lines.append("- Verify ticker symbols are correct")
                        elif stage == "strategy_setup":
                            lines.append("- Check strategy parameters for valid values")
                            lines.append("- Ensure strategy class is properly implemented")
                        elif stage == "parameter_grid_generation":
                            lines.append("- Verify parameter grid configuration")
                            lines.append("- Check for invalid parameter combinations")
                        elif stage == "optimization_iterations":
                            lines.append("- Check parameter ranges for potential division by zero")
                            lines.append("- Consider reducing the optimization space")
                        elif stage == "monte_carlo_execution":
                            lines.append("- Check for sufficient data for Monte Carlo simulations")
                            lines.append("- Consider reducing the number of simulations")
                        elif stage == "statistical_analysis":
                            lines.append("- Review trade data for statistical validity")
                            lines.append("- Ensure sufficient trades for meaningful analysis")
                        elif "backtest" in stage:
                            lines.append("- Check for proper strategy signal generation")
                            lines.append("- Verify position sizing calculations")
                        elif stage == "performance_evaluation":
                            lines.append("- Check for potential division by zero in metrics")
                            lines.append("- Verify that backtest results are properly formatted")
            else:
                lines.append("\nNo critical errors detected, but consider reviewing warnings.")
        
        return "\n".join(lines)
    
    def save_text_report(self, file_path: Optional[str] = None) -> str:
        """
        Save a text report to a file.
        
        Args:
            file_path: Path to save the report. If None, a default path is used.
            
        Returns:
            Path to the saved file
        """
        if not file_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.strategy_name}_{self.workflow_type}_stage_error_report_{timestamp}.txt"
            file_path = os.path.join(self.output_dir, file_name)
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Generate and save report
            report_text = self.generate_report_text()
            with open(file_path, 'w') as f:
                f.write(report_text)
                
            logger.info(f"Saved stage error text report to: {file_path}")
            return file_path
        except PermissionError:
            # If we can't write to the original path, try an alternative location
            alt_dir = os.path.join(project_root, "logs")
            os.makedirs(alt_dir, exist_ok=True)
            
            alt_path = os.path.join(alt_dir, os.path.basename(file_path))
            try:
                with open(alt_path, 'w') as f:
                    f.write(report_text)
                logger.warning(f"Permission denied for original path. Saved to: {alt_path}")
                return alt_path
            except Exception as e:
                logger.error(f"Failed to save report to alternative location: {e}")
                return "Error: Could not save report file"
        except Exception as e:
            logger.error(f"Error saving text report: {e}")
            return "Error: Could not save report file"
        
def detect_workflow_stage(log_line: str, workflow_type: str) -> Optional[str]:
    """
    Detect the workflow stage from a log line.
    
    Args:
        log_line: Line from a log file
        workflow_type: Type of workflow (simple, optimization, etc.)
        
    Returns:
        Detected stage name or None if not detected
    """
    # Get the stages for this workflow type
    if workflow_type not in WORKFLOW_STAGES:
        return None
        
    stages = WORKFLOW_STAGES[workflow_type]
    
    # Common stage indicators in log messages
    stage_indicators = {
        "initialization": ["initializing", "starting workflow", "setting up workflow", 
                          "beginning workflow", "workflow setup", "configuring workflow"],
        "data_loading": ["loading data", "preparing data", "fetching data", "reading data", 
                        "loading stock data", "data preparation"],
        "strategy_setup": ["setting up strategy", "initializing strategy", "loading strategy", 
                          "configuring strategy parameters", "strategy initialization"],
        "backtest_execution": ["running backtest", "executing strategy", "simulating trades", 
                              "processing signals", "running simulation"],
        "performance_evaluation": ["evaluating performance", "calculating metrics", 
                                  "computing statistics", "analyzing results"],
        "parameter_grid_generation": ["generating parameter grid", "creating grid", 
                                     "setting up parameter space", "parameter combinations"],
        "optimization_iterations": ["running optimization", "testing parameters", 
                                   "parameter trial", "optimization iteration"],
        "best_parameter_selection": ["selecting best parameters", "finding optimal parameters", 
                                    "best parameter set", "optimal configuration"],
        "backtest_with_best_params": ["running with best parameters", "optimal parameter backtest", 
                                     "validation backtest", "final backtest"],
        "initial_backtest": ["initial backtest", "base case backtest", "reference backtest"],
        "simulation_generation": ["generating simulations", "creating monte carlo simulations", 
                                 "permuting data", "bootstrap samples"],
        "monte_carlo_execution": ["running monte carlo", "executing simulations", 
                                 "monte carlo iterations", "simulation runs"],
        "statistical_analysis": ["analyzing statistics", "calculating p-values", 
                                "statistical validation", "confidence intervals"],
        "window_generation": ["creating windows", "generating time periods", 
                             "setting up walkforward windows", "time slices"],
        "in_sample_optimization": ["in-sample optimization", "training window", 
                                  "optimizing in-sample", "training period"],
        "out_of_sample_testing": ["out-of-sample testing", "validation window", 
                                 "testing period", "out-sample validation"],
        "combined_results_analysis": ["combining results", "merging window results", 
                                     "walkforward analysis", "window combination"],
        "simple_backtest": ["simple backtest", "initial strategy test", "baseline test"],
        "parameter_optimization": ["parameter optimization", "optimizing parameters", 
                                  "grid search", "parameter search"],
        "walkforward_analysis": ["walkforward analysis", "walk-forward testing", 
                                "time window validation", "rolling window test"],
        "monte_carlo_simulation": ["monte carlo simulation", "randomized testing", 
                                  "stochastic simulation", "statistical validation"],
        "optimized_backtest": ["optimized backtest", "best parameter test", 
                              "optimal configuration test", "final validation"]
    }
    
    # Search for stage indicators in the log line
    for stage in stages:
        if stage in stage_indicators:
            indicators = stage_indicators[stage]
            for indicator in indicators:
                if indicator.lower() in log_line.lower():
                    return stage
    
    # If no specific match, use more general detection
    if "data" in log_line.lower() and any(word in log_line.lower() for word in ["load", "read", "fetch", "prepare"]):
        return "data_loading"
        
    if "strategy" in log_line.lower() and any(word in log_line.lower() for word in ["setup", "init", "config"]):
        return "strategy_setup"
        
    if "backtest" in log_line.lower() and any(word in log_line.lower() for word in ["run", "execute", "process"]):
        return "backtest_execution"
        
    if "performance" in log_line.lower() or "metrics" in log_line.lower() or "statistics" in log_line.lower():
        return "performance_evaluation"
        
    # If still no match, try to infer from specific keywords
    if "parameter" in log_line.lower() and "grid" in log_line.lower():
        return "parameter_grid_generation"
        
    if "optimization" in log_line.lower() or "trial" in log_line.lower():
        return "optimization_iterations"
        
    if "monte carlo" in log_line.lower() or "simulation" in log_line.lower():
        return "monte_carlo_execution"
        
    if "window" in log_line.lower() or "walkforward" in log_line.lower():
        return "window_generation"
    
    # Still no match, return None
    return None
    
def extract_stage_errors(output_dir: str, workflow_type: str, strategy_name: str) -> StageErrorReport:
    """
    Extract stage-specific errors from workflow logs and results.
    
    Args:
        output_dir: Directory containing the workflow output
        workflow_type: Type of workflow
        strategy_name: Name of the strategy
        
    Returns:
        StageErrorReport containing all detected errors
    """
    # Create a new error report
    report = StageErrorReport(
        workflow_type=workflow_type,
        strategy_name=strategy_name,
        output_dir=output_dir
    )
    
    # List of error indicators to look for
    error_indicators = {
        "CRITICAL": ["critical", "exception", "traceback", "failed with exception", "runtime error",
                    "assertion error", "fatal error"],
        "ERROR": ["error", "failed", "failure", "incorrect", "invalid input", "unable to", 
                 "cannot", "division by zero"],
        "WARNING": ["warning", "possible issue", "potential problem", "unexpected", "may cause issues"]
    }
    
    # Find all log files
    log_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".log") or (file.endswith(".txt") and "summary" in file):
                log_files.append(os.path.join(root, file))
    
    # Process each log file
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            # Process each line
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Determine if this is an error line
                severity = None
                for sev, indicators in error_indicators.items():
                    if any(ind.lower() in line.lower() for ind in indicators):
                        severity = sev
                        break
                
                # If we found an error, create a StageError
                if severity:
                    # Try to detect the stage
                    stage = detect_workflow_stage(line, workflow_type)
                    
                    # If we couldn't detect the stage from this line,
                    # look at surrounding context lines
                    if not stage:
                        # Check previous lines for stage indicators
                        context_start = max(0, i - 5)
                        for j in range(context_start, i):
                            stage = detect_workflow_stage(lines[j], workflow_type)
                            if stage:
                                break
                                
                        # If still no stage, use "unknown"
                        if not stage:
                            stage = "unknown"
                    
                    # Get context for the error
                    context_before = []
                    context_after = []
                    
                    # Lines before the error
                    context_start = max(0, i - 3)
                    for j in range(context_start, i):
                        context_before.append(f"Line {j+1}: {lines[j].strip()}")
                    
                    # Lines after the error
                    context_end = min(len(lines), i + 4)
                    for j in range(i + 1, context_end):
                        context_after.append(f"Line {j+1}: {lines[j].strip()}")
                    
                    # Create and add the error
                    error = StageError(
                        message=line.strip(),
                        stage=stage,
                        severity=severity,
                        file_path=log_file,
                        line_number=i + 1,
                        context_before=context_before,
                        context_after=context_after
                    )
                    
                    report.add_error(error)
        
        except Exception as e:
            logger.error(f"Error processing log file {log_file}: {e}")
            # Add this as an error in the report
            report.add_exception(
                exception=e,
                stage="unknown",
                message=f"Error processing log file {os.path.basename(log_file)}: {e}"
            )
    
    # Also look for error information in summary files
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt") and "summary" in file.lower():
                summary_path = os.path.join(root, file)
                
                try:
                    with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.readlines()
                    
                    # Look for workflow status
                    workflow_status = None
                    for line in content:
                        if "WORKFLOW STATUS:" in line:
                            workflow_status = line.strip()
                            break
                    
                    # If workflow failed, record this as an error
                    if workflow_status and "ERROR" in workflow_status:
                        # Create an error for this
                        error = StageError(
                            message=workflow_status,
                            stage="unknown",  # We don't know which stage failed
                            severity="ERROR",
                            file_path=summary_path
                        )
                        report.add_error(error)
                        
                    # Also look for specific error sections
                    in_error_section = False
                    error_context = []
                    
                    for i, line in enumerate(content):
                        if "Error Information:" in line or "Error:" in line:
                            in_error_section = True
                            error_context = []
                            continue
                            
                        if in_error_section:
                            # If we hit a new section, end the error section
                            if line.startswith("===") or line.startswith("---"):
                                in_error_section = False
                                
                                # Process the collected error context
                                if error_context:
                                    # Join the error context into a single message
                                    error_msg = " ".join(error_context)
                                    
                                    # Try to detect the stage
                                    stage = "unknown"
                                    for ctx_line in error_context:
                                        detected_stage = detect_workflow_stage(ctx_line, workflow_type)
                                        if detected_stage:
                                            stage = detected_stage
                                            break
                                    
                                    # Create the error
                                    error = StageError(
                                        message=error_msg,
                                        stage=stage,
                                        severity="ERROR",
                                        file_path=summary_path
                                    )
                                    report.add_error(error)
                                    
                                # Reset for next error section
                                error_context = []
                            else:
                                # Add line to error context
                                error_context.append(line.strip())
                                
                except Exception as e:
                    logger.error(f"Error processing summary file {summary_path}: {e}")
    
    # Return the completed report
    return report

def create_stage_error_report(output_dir: str, workflow_type: str, strategy_name: str) -> str:
    """
    Create and save a stage-specific error report for a workflow run.
    
    Args:
        output_dir: Directory containing the workflow output
        workflow_type: Type of workflow
        strategy_name: Name of the strategy
        
    Returns:
        Path to the saved report file
    """
    # Extract errors
    report = extract_stage_errors(output_dir, workflow_type, strategy_name)
    
    # Save both JSON and text versions of the report
    json_path = report.save()
    text_path = report.save_text_report()
    
    # Log report generation
    if report.has_errors():
        if report.has_critical_errors():
            logger.warning(f"Created stage error report with {len(report.get_all_errors())} errors "
                         f"including CRITICAL errors: {text_path}")
        else:
            logger.info(f"Created stage error report with {len(report.get_all_errors())} errors: {text_path}")
    else:
        logger.info(f"Created stage error report with no errors: {text_path}")
    
    return text_path

def generate_error_summary(output_dir: str, workflow_type: str, strategy_name: str, 
                          start_date: str, end_date: str, tickers: List[str], 
                          error_msg: str = None) -> str:
    """
    Generate a comprehensive error summary for a workflow run.
    
    Args:
        output_dir: Directory where workflow results are stored
        workflow_type: Type of workflow (simple, monte_carlo, etc.)
        strategy_name: Name of the strategy
        start_date: Start date for the backtest
        end_date: End date for the backtest
        tickers: List of ticker symbols
        error_msg: Optional specific error message to include
        
    Returns:
        Path to the generated error summary file
    """
    logger.info(f"Generating error summary for {workflow_type} workflow with {strategy_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to save the error summary
    summary_file = os.path.join(output_dir, "error_summary.txt")
    
    # Collect errors from logs
    from workflows.workflow_utils import check_logs_for_errors, print_error_report
    error_logs = check_logs_for_errors(output_dir)
    
    # Load config and parameters if available
    parameters = {}
    config_file = None
    
    # Look for a config.json file
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and 'config' in filename.lower():
            config_file = os.path.join(output_dir, filename)
            break
    
    # If we found a config file, load it
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Extract parameters for the strategy
            if 'strategies' in config and strategy_name in config['strategies']:
                parameters = config['strategies'][strategy_name].get('parameters', {})
            elif 'parameters' in config:
                parameters = config['parameters']
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
    
    # Write the summary file
    with open(summary_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"{workflow_type.upper()} ERROR SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic test information
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        if isinstance(tickers, list):
            f.write(f"Tickers: {', '.join(tickers)}\n")
        else:
            f.write(f"Tickers: {tickers}\n")
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Error Report Generated: {timestamp}\n\n")
        
        # Parameters section
        f.write("-" * 80 + "\n")
        f.write("Parameters:\n")
        f.write("-" * 80 + "\n")
        if parameters:
            for param, value in parameters.items():
                f.write(f"{param}: {value}\n")
        else:
            f.write("No parameters found.\n")
        f.write("\n")
        
        # Error information
        f.write("-" * 80 + "\n")
        f.write("Error Information:\n")
        f.write("-" * 80 + "\n")
        
        # Include specific error message if provided
        if error_msg:
            f.write(f"Primary Error: {error_msg}\n\n")
        
        # Error summary statistics
        if not error_logs:
            f.write("No additional errors found in log files.\n")
        else:
            # Count errors by severity
            critical_count = 0
            error_count = 0
            total_errors = 0
            files_with_errors = len(error_logs)
            
            for log_file, errors in error_logs.items():
                for error in errors:
                    total_errors += 1
                    if error['severity'] == 'CRITICAL':
                        critical_count += 1
                    else:
                        error_count += 1
            
            # Add error statistics
            f.write(f"Total errors found: {total_errors} in {files_with_errors} files\n")
            f.write(f"Critical errors: {critical_count}\n")
            f.write(f"Regular errors: {error_count}\n\n")
            
            # List errors by file
            for log_file, errors in error_logs.items():
                # Get relative path from output directory
                try:
                    rel_path = os.path.relpath(log_file, output_dir)
                    display_path = rel_path
                except:
                    display_path = os.path.basename(log_file)
                    
                f.write(f"File: {display_path}\n")
                f.write("-" * 40 + "\n")
                
                # List errors (simplified for summary)
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. {error['message']}\n")
                
                f.write("\n")
        
        # End of summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("WORKFLOW STATUS: ERROR\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Error summary saved to: {summary_file}")
    
    # Generate a more detailed error report with context
    error_report_path = os.path.join(output_dir, "error_report.txt")
    print_error_report(error_logs, error_report_path)
    
    return summary_file

def collect_workflow_errors(output_root: str = None, days: int = 7) -> Dict[str, Any]:
    """
    Collect errors from all workflow runs within a specified timeframe.
    
    Args:
        output_root: Root directory containing workflow outputs
        days: Only include errors from the last N days
        
    Returns:
        Dictionary with workflow statistics and error summaries
    """
    if output_root is None:
        output_root = os.path.join(project_root, "output")
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Results dictionary
    results = {
        "total_workflow_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "workflows_with_errors": [],
        "error_counts": {
            "critical": 0,
            "error": 0,
            "warning": 0
        },
        "errors_by_workflow": {},
        "errors_by_strategy": {},
        "most_common_errors": []
    }
    
    # Dictionary to track error occurrences
    error_occurrences = {}
    
    # Walk through output directories
    for root, dirs, files in os.walk(output_root):
        # Check if this is a workflow output directory
        if any(f.endswith('_summary.txt') for f in files) or any(f.endswith('.log') for f in files):
            # Get directory modification time
            try:
                mod_time = os.path.getmtime(root)
                dir_date = datetime.datetime.fromtimestamp(mod_time)
                if dir_date < cutoff_date:
                    continue  # Skip older directories
            except Exception:
                pass  # If we can't determine date, include it
            
            # Try to determine workflow type and strategy from directory name
            dir_name = os.path.basename(root)
            strategy_name = None
            workflow_type = None
            
            # Parse directory name (format: Strategy_workflow_timestamp_id)
            parts = dir_name.split('_')
            if len(parts) >= 2:
                strategy_name = parts[0]
                for workflow in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
                    if workflow in dir_name.lower():
                        workflow_type = workflow
                        break
            
            # Count this as a workflow run
            results["total_workflow_runs"] += 1
            
            # Check for error files or logs
            has_error = False
            error_files = []
            
            for filename in files:
                if filename == 'error_summary.txt' or filename == 'error_report.txt':
                    has_error = True
                    error_files.append(os.path.join(root, filename))
                elif '_summary.txt' in filename:
                    # Check if the summary file indicates an error
                    try:
                        with open(os.path.join(root, filename), 'r') as f:
                            content = f.read()
                            if "WORKFLOW STATUS: ERROR" in content or "Error:" in content:
                                has_error = True
                                error_files.append(os.path.join(root, filename))
                    except Exception:
                        pass
            
            # Update counts
            if has_error:
                results["failed_runs"] += 1
                
                # Create workflow entry
                workflow_entry = {
                    "path": root,
                    "strategy": strategy_name,
                    "workflow_type": workflow_type,
                    "timestamp": dir_date.strftime("%Y-%m-%d %H:%M:%S") if 'dir_date' in locals() else "Unknown",
                    "error_files": error_files
                }
                
                # Extract key errors
                key_errors = []
                for error_file in error_files:
                    try:
                        with open(error_file, 'r') as f:
                            content = f.read()
                            
                            # Extract main errors using regex
                            error_matches = re.findall(r'Error:.*?$(.*?$)?', content, re.MULTILINE)
                            key_errors.extend([e.strip() for e in error_matches if e.strip()])
                            
                            # Extract critical errors
                            critical_matches = re.findall(r'\[CRITICAL\].*?$', content, re.MULTILINE)
                            key_errors.extend([e.strip() for e in critical_matches if e.strip()])
                            
                            # Count errors by type
                            results["error_counts"]["critical"] += content.count("[CRITICAL]")
                            results["error_counts"]["error"] += content.count("[ERROR]")
                            results["error_counts"]["warning"] += content.count("[WARNING]")
                            
                            # Track error occurrences for most common errors
                            for error_line in content.split('\n'):
                                if "[CRITICAL]" in error_line or "[ERROR]" in error_line:
                                    # Normalize error message (remove timestamps, line numbers, variables)
                                    norm_error = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', error_line)
                                    norm_error = re.sub(r'line \d+', 'line X', norm_error)
                                    norm_error = re.sub(r':\d+:', ':X:', norm_error)
                                    
                                    if norm_error in error_occurrences:
                                        error_occurrences[norm_error] += 1
                                    else:
                                        error_occurrences[norm_error] = 1
                    except Exception as e:
                        key_errors.append(f"Error reading file: {e}")
                
                workflow_entry["key_errors"] = key_errors
                results["workflows_with_errors"].append(workflow_entry)
                
                # Update errors by workflow and strategy
                if workflow_type:
                    if workflow_type not in results["errors_by_workflow"]:
                        results["errors_by_workflow"][workflow_type] = 1
                    else:
                        results["errors_by_workflow"][workflow_type] += 1
                
                if strategy_name:
                    if strategy_name not in results["errors_by_strategy"]:
                        results["errors_by_strategy"][strategy_name] = 1
                    else:
                        results["errors_by_strategy"][strategy_name] += 1
            else:
                results["successful_runs"] += 1
    
    # Calculate most common errors
    most_common = sorted(error_occurrences.items(), key=lambda x: x[1], reverse=True)
    results["most_common_errors"] = [{"error": e, "count": c} for e, c in most_common[:10]]
    
    # Calculate success rate
    if results["total_workflow_runs"] > 0:
        results["success_rate"] = (results["successful_runs"] / results["total_workflow_runs"]) * 100
    else:
        results["success_rate"] = 0
    
    return results

def generate_consolidated_error_report(output_dir: str = None, days: int = 7) -> str:
    """
    Generate a consolidated error report for all workflow runs.
    
    Args:
        output_dir: Directory to save the report
        days: Only include errors from the last N days
        
    Returns:
        Path to the generated report file
    """
    # If no specific output directory is provided, use the current directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"consolidated_error_report_{timestamp}.txt")
    
    # Collect errors
    error_data = collect_workflow_errors(days=days)
    
    # Write the report
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONSOLIDATED ERROR REPORT - Last {days} days\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total workflow runs: {error_data['total_workflow_runs']}\n")
        f.write(f"Successful runs: {error_data['successful_runs']}\n")
        f.write(f"Failed runs: {error_data['failed_runs']}\n")
        
        if error_data['total_workflow_runs'] > 0:
            success_rate = (error_data['successful_runs'] / error_data['total_workflow_runs']) * 100
            f.write(f"Success rate: {success_rate:.2f}%\n\n")
        
        # Error counts
        f.write("ERROR COUNTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Critical errors: {error_data['error_counts']['critical']}\n")
        f.write(f"Regular errors: {error_data['error_counts']['error']}\n")
        f.write(f"Warnings: {error_data['error_counts']['warning']}\n\n")
        
        # Errors by workflow type
        f.write("ERRORS BY WORKFLOW TYPE\n")
        f.write("-" * 80 + "\n")
        for workflow, count in error_data['errors_by_workflow'].items():
            f.write(f"{workflow}: {count}\n")
        f.write("\n")
        
        # Errors by strategy
        f.write("ERRORS BY STRATEGY\n")
        f.write("-" * 80 + "\n")
        for strategy, count in error_data['errors_by_strategy'].items():
            f.write(f"{strategy}: {count}\n")
        f.write("\n")
        
        # Most common errors
        f.write("MOST COMMON ERRORS\n")
        f.write("-" * 80 + "\n")
        for i, error_info in enumerate(error_data['most_common_errors'], 1):
            f.write(f"{i}. [{error_info['count']} occurrences] {error_info['error']}\n")
        f.write("\n")
        
        # Individual workflow errors
        f.write("INDIVIDUAL WORKFLOW ERRORS\n")
        f.write("-" * 80 + "\n")
        for i, workflow in enumerate(error_data['workflows_with_errors'], 1):
            f.write(f"Workflow #{i}: {workflow['strategy']} - {workflow['workflow_type']}\n")
            f.write(f"Path: {workflow['path']}\n")
            f.write(f"Timestamp: {workflow['timestamp']}\n")
            
            if workflow['key_errors']:
                f.write("Key errors:\n")
                for j, error in enumerate(workflow['key_errors'], 1):
                    f.write(f"  {j}. {error}\n")
            else:
                f.write("No key errors extracted.\n")
            
            f.write("-" * 40 + "\n\n")
        
        # End of report
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Consolidated error report saved to: {report_path}")
    return report_path

def create_consolidated_stage_error_report(output_root: str = None, days: int = 7) -> str:
    """
    Create a consolidated stage error report for multiple workflow runs.
    
    Args:
        output_root: Root directory containing workflow outputs
        days: Only include workflow runs from the last N days
        
    Returns:
        Path to the saved consolidated report file
    """
    if output_root is None:
        output_root = os.path.join(project_root, "output")
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Find all workflow output directories
    workflow_dirs = []
    all_reports = []
    
    # Walk through output directories
    for root, dirs, files in os.walk(output_root):
        # Check if this is a workflow output directory
        if any(f.endswith('_summary.txt') for f in files) or any(f.endswith('.log') for f in files):
            # Get directory modification time
            try:
                mod_time = os.path.getmtime(root)
                dir_date = datetime.datetime.fromtimestamp(mod_time)
                if dir_date < cutoff_date:
                    continue  # Skip older directories
            except:
                pass  # If we can't determine date, include it
            
            # Try to determine workflow type and strategy from directory name
            dir_name = os.path.basename(root)
            strategy_name = "unknown"
            workflow_type = "unknown"
            
            # Parse directory name (format: Strategy_workflow_timestamp_id)
            parts = dir_name.split('_')
            if len(parts) >= 2:
                strategy_name = parts[0]
                for workflow in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
                    if workflow in dir_name.lower():
                        workflow_type = workflow
                        break
            
            # Add to list of directories to process
            workflow_dirs.append({
                "path": root,
                "strategy": strategy_name,
                "workflow": workflow_type,
                "timestamp": dir_date.strftime("%Y-%m-%d %H:%M:%S") if 'dir_date' in locals() else "Unknown"
            })
    
    # Process each workflow directory
    for workflow_dir in workflow_dirs:
        # Extract stage errors
        report = extract_stage_errors(
            workflow_dir["path"],
            workflow_dir["workflow"],
            workflow_dir["strategy"]
        )
        
        # Only keep reports with errors
        if report.has_errors():
            all_reports.append(report)
            
            # Save individual report
            report.save_text_report()
    
    # Create consolidated report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    consolidated_path = os.path.join(output_root, f"consolidated_stage_error_report_{timestamp}.txt")
    
    with open(consolidated_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONSOLIDATED STAGE ERROR REPORT - Last {days} days\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        total_errors = sum(len(report.get_all_errors()) for report in all_reports)
        critical_errors = sum(1 for report in all_reports if report.has_critical_errors())
        
        f.write(f"Total workflow runs analyzed: {len(workflow_dirs)}\n")
        f.write(f"Workflow runs with errors: {len(all_reports)}\n")
        f.write(f"Total errors found: {total_errors}\n")
        f.write(f"Workflow runs with critical errors: {critical_errors}\n\n")
        
        # Error counts by stage
        stage_counts = defaultdict(int)
        for report in all_reports:
            for stage, count in report.count_errors_by_stage().items():
                stage_counts[stage] += count
        
        f.write("ERRORS BY STAGE\n")
        f.write("-" * 80 + "\n")
        
        # Sort stages by error count
        sorted_stages = sorted(stage_counts.items(), key=lambda x: x[1], reverse=True)
        for stage, count in sorted_stages:
            stage_display = stage.replace('_', ' ').title()
            f.write(f"{stage_display}: {count}\n")
        f.write("\n")
        
        # Error counts by severity
        severity_counts = defaultdict(int)
        for report in all_reports:
            for severity, count in report.count_errors_by_severity().items():
                severity_counts[severity] += count
        
        f.write("ERRORS BY SEVERITY\n")
        f.write("-" * 80 + "\n")
        for severity, count in sorted(severity_counts.items(), 
                                     key=lambda x: ERROR_SEVERITY.get(x[0], 0), 
                                     reverse=True):
            f.write(f"{severity}: {count}\n")
        f.write("\n")
        
        # Workflow runs with critical errors
        critical_workflows = [r for r in all_reports if r.has_critical_errors()]
        if critical_workflows:
            f.write("WORKFLOW RUNS WITH CRITICAL ERRORS\n")
            f.write("-" * 80 + "\n")
            for i, report in enumerate(critical_workflows, 1):
                f.write(f"{i}. {report.strategy_name} ({report.workflow_type}) - {len(report.get_all_errors())} errors\n")
                f.write(f"   Directory: {report.output_dir}\n")
                
                # List critical errors
                critical_errors = [e for e in report.get_all_errors() if e.severity == "CRITICAL"]
                for j, error in enumerate(critical_errors[:3], 1):  # Show up to 3 critical errors
                    f.write(f"   {j}. {error.stage}: {error.message[:100]}...\n")
                
                if len(critical_errors) > 3:
                    f.write(f"   ... and {len(critical_errors) - 3} more critical errors\n")
                    
                f.write("\n")
        
        # Detailed per-workflow summaries (limit to 10 most recent)
        recent_reports = sorted(all_reports, key=lambda r: r.updated_at, reverse=True)[:10]
        
        f.write("DETAILED ERROR SUMMARIES (10 most recent)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, report in enumerate(recent_reports, 1):
            f.write(f"Workflow {i}: {report.strategy_name} ({report.workflow_type})\n")
            f.write(f"Directory: {report.output_dir}\n")
            f.write(f"Total errors: {len(report.get_all_errors())}\n")
            
            # Errors by stage for this workflow
            f.write("\nErrors by stage:\n")
            for stage, count in report.count_errors_by_stage().items():
                stage_display = stage.replace('_', ' ').title()
                f.write(f"  {stage_display}: {count}\n")
            
            # Most severe errors for this workflow
            all_errors = report.get_all_errors()
            sorted_errors = sorted(all_errors, 
                                  key=lambda e: (e.severity_level, e.timestamp), 
                                  reverse=True)
            
            f.write("\nMost severe errors:\n")
            for j, error in enumerate(sorted_errors[:5], 1):  # Show up to 5 errors
                f.write(f"  {j}. [{error.severity}] {error.stage}: {error.message[:100]}" + 
                      ("..." if len(error.message) > 100 else "") + "\n")
            
            if len(sorted_errors) > 5:
                f.write(f"  ... and {len(sorted_errors) - 5} more errors\n")
                
            f.write("\n" + "-" * 80 + "\n\n")
    
    logger.info(f"Consolidated stage error report saved to: {consolidated_path}")
    return consolidated_path

def update_workflow_with_stage_error_reporting(module_path: str) -> bool:
    """
    Update a workflow module to include stage error reporting.
    
    Args:
        module_path: Path to the workflow module file
        
    Returns:
        True if the module was updated successfully, False otherwise
    """
    try:
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Check if already has stage error reporting
        if "create_stage_error_report" in content:
            logger.info(f"Module {module_path} already has stage error reporting")
            return True
        
        # Look for the run_* function (main entry point)
        import re
        
        # Common patterns for run_* functions in workflow modules
        run_patterns = [
            r'def\s+run_(\w+)_workflow\s*\(',  # run_simple_workflow, run_optimization_workflow, etc.
            r'def\s+run_workflow\s*\('          # Generic run_workflow function
        ]
        
        # Find a matching run function
        run_match = None
        for pattern in run_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                run_match = match
                break
            if run_match:
                break
        
        if not run_match:
            logger.warning(f"Could not find run_* function in {module_path}")
            return False
        
        # Detect workflow type from filename or function name
        workflow_type = "unknown"
        filename = os.path.basename(module_path)
        if "_workflow.py" in filename:
            workflow_type = filename.replace("_workflow.py", "")
        elif run_match and run_match.group(1):
            workflow_type = run_match.group(1)
        
        # Import statement to add
        import_stmt = "\nfrom utils.error_reporting import create_stage_error_report, StageError, StageErrorReport"
        
        # Check if we need to add the import
        if "from utils.error_reporting import" not in content:
            # Find where to add the import - after other imports
            import_section_end = 0
            import_lines = re.finditer(r'^(?:import|from)\s+.*$', content, re.MULTILINE)
            for match in import_lines:
                import_section_end = max(import_section_end, match.end())
            
            if import_section_end > 0:
                # Add import after the last import statement
                new_content = (content[:import_section_end] + import_stmt + 
                              content[import_section_end:])
                content = new_content
        
        # Find the try-except block around the main function execution
        try_except_patterns = [
            r'(\s+)try:.*?except\s+Exception\s+as\s+e:.*?return\s+(?:.*?)$',  # Common try-except with return
            r'(\s+)try:.*?except\s+Exception\s+as\s+e:.*?}$'                   # Try-except with dict return
        ]
        
        try_except_match = None
        for pattern in try_except_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
            for match in matches:
                try_except_match = match
                break
            if try_except_match:
                break
        
        if not try_except_match:
            logger.warning(f"Could not find try-except block in {module_path}")
            return False
        
        # Find where to add the error reporting - in the except block
        except_block = re.search(r'(\s+)except\s+Exception\s+as\s+e:(.*?)(?:return|$)', 
                               content, re.DOTALL)
        
        if not except_block:
            logger.warning(f"Could not find except block in {module_path}")
            return False
        
        # Indent level from the match
        indent = except_block.group(1)
        
        # Generate error reporting code to add
        error_report_code = f"\n{indent}    # Generate stage error report\n"
        error_report_code += f"{indent}    try:\n"
        error_report_code += f"{indent}        create_stage_error_report(output_dir, '{workflow_type}', strategy_name)\n"
        error_report_code += f"{indent}    except Exception as report_err:\n"
        error_report_code += f"{indent}        logger.error(f\"Error generating stage error report: {{report_err}}\")\n"
        
        # Find the right position to insert in the except block
        except_content = except_block.group(2)
        insertion_point = except_block.start(2) + len(except_content)
        
        # Insert the error reporting code
        new_content = content[:insertion_point] + error_report_code + content[insertion_point:]
        
        # Write the updated content back to the file
        with open(module_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added stage error reporting to {module_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {module_path} with stage error reporting: {e}")
        return False

def update_all_workflow_modules():
    """Update all workflow modules with stage error reporting."""
    # List of workflow modules to update
    workflow_dir = os.path.join(project_root, "src", "workflows")
    
    # Find all workflow modules
    workflow_modules = []
    for file in os.listdir(workflow_dir):
        if file.endswith("_workflow.py") or file in ["unified_workflow.py", "complete_workflow.py"]:
            workflow_modules.append(os.path.join(workflow_dir, file))
    
    # Update each module
    success_count = 0
    for module_path in workflow_modules:
        if update_workflow_with_stage_error_reporting(module_path):
            success_count += 1
    
    logger.info(f"Updated {success_count}/{len(workflow_modules)} workflow modules with stage error reporting")
    return success_count

def main():
    """Command-line interface for error reporting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced error reports for workflows")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include in the report")
    parser.add_argument("--output-dir", type=str, help="Directory to save the report")
    parser.add_argument("--workflow-dir", type=str, help="Specific workflow directory to analyze")
    parser.add_argument("--update-modules", action="store_true", help="Update workflow modules with stage error reporting")
    parser.add_argument("--stage-report", action="store_true", help="Generate a stage-specific error report")
    parser.add_argument("--workflow-type", type=str, help="Workflow type (for stage report)")
    parser.add_argument("--strategy", type=str, help="Strategy name (for stage report)")
    args = parser.parse_args()
    
    if args.update_modules:
        # Update workflow modules with stage error reporting
        success_count = update_all_workflow_modules()
        print(f"Updated {success_count} workflow modules with stage error reporting")
        return
    
    if args.workflow_dir:
        # Generate an error report for a specific workflow directory
        if not os.path.exists(args.workflow_dir):
            print(f"Error: Directory {args.workflow_dir} does not exist")
            return
            
        # Try to determine workflow type and strategy
        dir_name = os.path.basename(args.workflow_dir)
        parts = dir_name.split('_')
        
        strategy_name = args.strategy or (parts[0] if len(parts) > 0 else "Unknown")
        
        workflow_type = args.workflow_type or "unknown"
        if not args.workflow_type:
            for wf in ['simple', 'monte_carlo', 'optimization', 'walkforward', 'complete']:
                if wf in dir_name.lower():
                    workflow_type = wf
                    break
        
        if args.stage_report:
            # Generate a stage-specific error report
            report_path = create_stage_error_report(
                args.workflow_dir,
                workflow_type,
                strategy_name
            )
            print(f"Stage error report generated: {report_path}")
        else:
            # Generate a traditional error summary
            summary_file = generate_error_summary(
                args.workflow_dir, 
                workflow_type,
                strategy_name,
                "Unknown", "Unknown",
                ["Unknown"],
                "Manual error report generation"
            )
            print(f"Error summary generated: {summary_file}")
    else:
        # Generate a consolidated report
        if args.stage_report:
            # Generate a consolidated stage error report
            report_path = create_consolidated_stage_error_report(
                args.output_dir,
                args.days
            )
            print(f"Consolidated stage error report generated: {report_path}")
        else:
            # Generate a traditional consolidated error report
            report_path = generate_consolidated_error_report(
                args.output_dir,
                args.days
            )
            print(f"Consolidated error report generated: {report_path}")

if __name__ == "__main__":
    main()