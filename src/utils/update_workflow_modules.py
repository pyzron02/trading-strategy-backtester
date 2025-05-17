#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to update all workflow modules with stage error reporting.

This script finds all workflow modules in the project and updates them
to include stage-specific error reporting functionality. It can be run
as a standalone script during development or when updating existing modules.

Usage:
    python update_workflow_modules.py [--dry-run] [--workflow-dir DIR]

Options:
    --dry-run        Show changes that would be made without actually making them
    --workflow-dir   Path to the workflow directory (default: src/workflows)
"""
import os
import sys
import re
import argparse
import shutil
from typing import List, Dict, Any, Optional, Tuple

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the error reporting functionality
from utils.error_reporting import update_workflow_with_stage_error_reporting

def find_workflow_modules(workflow_dir: str) -> List[str]:
    """
    Find all workflow module files in the specified directory.
    
    Args:
        workflow_dir: Path to the workflow directory
        
    Returns:
        List of workflow module file paths
    """
    workflow_modules = []
    
    if not os.path.exists(workflow_dir):
        print(f"Error: Workflow directory {workflow_dir} does not exist")
        return []
    
    # Find all Python files that look like workflows
    for filename in os.listdir(workflow_dir):
        if not filename.endswith('.py'):
            continue
            
        if (filename.endswith('_workflow.py') or 
            filename == 'unified_workflow.py' or 
            filename == 'complete_workflow.py' or
            'workflow' in filename.lower()):
            
            workflow_modules.append(os.path.join(workflow_dir, filename))
    
    return workflow_modules

def create_backup(file_path: str) -> str:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to back up
        
    Returns:
        Path to the backup file
    """
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    return backup_path

def restore_from_backup(backup_path: str) -> bool:
    """
    Restore a file from its backup.
    
    Args:
        backup_path: Path to the backup file
        
    Returns:
        True if the restore was successful, False otherwise
    """
    original_path = backup_path.replace('.bak', '')
    try:
        shutil.copy2(backup_path, original_path)
        return True
    except Exception as e:
        print(f"Error restoring from backup: {e}")
        return False

def detect_workflow_type(file_path: str) -> str:
    """
    Attempt to detect the workflow type from the file path or content.
    
    Args:
        file_path: Path to the workflow module
        
    Returns:
        Detected workflow type or "unknown"
    """
    # Extract from filename
    filename = os.path.basename(file_path)
    if filename.endswith('_workflow.py'):
        workflow_type = filename.replace('_workflow.py', '')
        return workflow_type
    
    # Extract from file content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for workflow type in function definitions
        match = re.search(r'def\s+run_(\w+)_workflow', content)
        if match:
            return match.group(1)
            
        # Check for common workflow types in the content
        for workflow_type in ['simple', 'optimization', 'monte_carlo', 'walkforward', 'complete']:
            if f'"{workflow_type}"' in content or f"'{workflow_type}'" in content:
                return workflow_type
    except Exception:
        pass
    
    return "unknown"

def analyze_module(file_path: str) -> Dict[str, Any]:
    """
    Analyze a workflow module file to determine if it needs updating.
    
    Args:
        file_path: Path to the workflow module
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file_path": file_path,
        "workflow_type": detect_workflow_type(file_path),
        "needs_update": False,
        "has_error_reporting": False,
        "has_try_except": False,
        "import_position": -1,
        "except_block_position": -1
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.readlines()
            
        # Check if already has error reporting
        for i, line in enumerate(content):
            if "create_stage_error_report" in line or "StageErrorReport" in line:
                result["has_error_reporting"] = True
                break
                
        # Find the import section
        import_end_line = 0
        for i, line in enumerate(content):
            if line.startswith('import ') or line.startswith('from '):
                import_end_line = i
        
        result["import_position"] = import_end_line + 1
        
        # Look for try-except blocks
        try_except_pattern = re.compile(r'^\s+try:')
        except_pattern = re.compile(r'^\s+except\s+.*:')
        
        for i, line in enumerate(content):
            if try_except_pattern.match(line):
                # Found a try block, look for the matching except
                for j in range(i + 1, len(content)):
                    if except_pattern.match(content[j]):
                        result["has_try_except"] = True
                        result["except_block_position"] = j
                        break
                if result["has_try_except"]:
                    break
        
        # Determine if update is needed
        result["needs_update"] = not result["has_error_reporting"] and result["has_try_except"]
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return result

def update_module(file_path: str, dry_run: bool = False) -> bool:
    """
    Update a workflow module with stage error reporting.
    
    Args:
        file_path: Path to the workflow module
        dry_run: If True, show changes without making them
        
    Returns:
        True if update was successful or would be successful, False otherwise
    """
    analysis = analyze_module(file_path)
    
    if not analysis["needs_update"]:
        if analysis["has_error_reporting"]:
            print(f"[SKIP] {os.path.basename(file_path)} - Already has error reporting")
        elif not analysis["has_try_except"]:
            print(f"[SKIP] {os.path.basename(file_path)} - No try-except block found")
        return False
    
    print(f"[{'WOULD UPDATE' if dry_run else 'UPDATE'}] {os.path.basename(file_path)} - {analysis['workflow_type']} workflow")
    
    if dry_run:
        return True
    
    # Create a backup
    backup_path = create_backup(file_path)
    print(f"  Created backup: {os.path.basename(backup_path)}")
    
    # Perform the update
    try:
        result = update_workflow_with_stage_error_reporting(file_path)
        if result:
            print(f"  Update successful ✅")
            return True
        else:
            print(f"  Update failed ❌")
            print(f"  Restoring from backup...")
            restore_from_backup(backup_path)
            return False
    except Exception as e:
        print(f"  Error during update: {e}")
        print(f"  Restoring from backup...")
        restore_from_backup(backup_path)
        return False

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(
        description="Update workflow modules with stage error reporting"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--workflow-dir", 
        type=str, 
        default=os.path.join(src_dir, "workflows"),
        help="Path to the workflow directory"
    )
    args = parser.parse_args()
    
    # Find workflow modules
    workflow_modules = find_workflow_modules(args.workflow_dir)
    
    if not workflow_modules:
        print(f"No workflow modules found in {args.workflow_dir}")
        return
    
    print(f"Found {len(workflow_modules)} workflow modules:")
    for module in workflow_modules:
        print(f"  - {os.path.basename(module)}")
    print()
    
    # Update each module
    success_count = 0
    for module in workflow_modules:
        if update_module(module, args.dry_run):
            success_count += 1
    
    # Print summary
    if args.dry_run:
        print(f"\nDRY RUN: Would have updated {success_count} of {len(workflow_modules)} modules")
    else:
        print(f"\nSuccessfully updated {success_count} of {len(workflow_modules)} modules")

if __name__ == "__main__":
    main()