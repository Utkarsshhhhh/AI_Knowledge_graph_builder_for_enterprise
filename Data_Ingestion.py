
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import logging

# Import configuration
try:
    from Config import BASE_DIR, DATA_FOLDER, INGESTED_DATA_PATH
except ImportError:
    print("⚠️ Config.py not found, using current directory")
    BASE_DIR = Path.cwd()
    DATA_FOLDER = BASE_DIR / "Data"
    INGESTED_DATA_PATH = BASE_DIR / "ingested_data.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FILE PROCESSORS
# ============================================================================

def process_csv_file(file_path: Path) -> Dict[str, Any]:
    """Process CSV file"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        
        if df.empty:
            logger.warning(f"Empty CSV file: {file_path.name}")
            return None
        
        # Convert datetime columns to string
        df = df.fillna("")
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        return {
            "file": file_path.name,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict("records")
        }
    except Exception as e:
        logger.error(f"Error processing CSV {file_path.name}: {e}")
        return None

def process_excel_file(file_path: Path) -> Dict[str, Any]:
    """Process Excel file"""
    try:
        df = pd.read_excel(file_path)
        
        if df.empty:
            logger.warning(f"Empty Excel file: {file_path.name}")
            return None
        
        # Convert datetime columns to string
        df = df.fillna("")
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        return {
            "file": file_path.name,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict("records")
        }
    except Exception as e:
        logger.error(f"Error processing Excel {file_path.name}: {e}")
        return None

def process_json_file(file_path: Path) -> Dict[str, Any]:
    """Process JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        return {
            "file": file_path.name,
            "type": type(json_data).__name__,
            "data": json_data
        }
    except Exception as e:
        logger.error(f"Error processing JSON {file_path.name}: {e}")
        return None

def process_pdf_file(file_path: Path) -> Dict[str, Any]:
    """Process PDF file"""
    try:
        import PyPDF2
    except ImportError:
        logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        if not text.strip():
            logger.warning(f"No extractable text in PDF: {file_path.name}")
            return None
        
        return {
            "file": file_path.name,
            "pages": len(reader.pages),
            "text_length": len(text),
            "data": text
        }
    except Exception as e:
        logger.error(f"Error processing PDF {file_path.name}: {e}")
        return None

def process_text_file(file_path: Path) -> Dict[str, Any]:
    """Process text file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            logger.warning(f"Empty text file: {file_path.name}")
            return None
        
        return {
            "file": file_path.name,
            "text_length": len(text),
            "data": text
        }
    except Exception as e:
        logger.error(f"Error processing text {file_path.name}: {e}")
        return None

# ============================================================================
# FILE DISPATCHER
# ============================================================================

def process_single_file(file_path: Path) -> tuple:
    """
    Process a single file based on its extension
    
    Returns:
        tuple: (category, result_dict)
    """
    ext = file_path.suffix.lower()
    
    # Skip output files
    if file_path.name == "ingested_data.json":
        return None, None
    
    logger.info(f"Processing: {file_path.name}")
    
    # Route to appropriate processor
    if ext == ".csv":
        result = process_csv_file(file_path)
        return ("structured", result) if result else (None, None)
    
    elif ext in [".xlsx", ".xls"]:
        result = process_excel_file(file_path)
        return ("structured", result) if result else (None, None)
    
    elif ext == ".json":
        result = process_json_file(file_path)
        return ("semi_structured", result) if result else (None, None)
    
    elif ext == ".pdf":
        result = process_pdf_file(file_path)
        return ("unstructured", result) if result else (None, None)
    
    elif ext in [".txt", ".log", ".md"]:
        result = process_text_file(file_path)
        return ("unstructured", result) if result else (None, None)
    
    else:
        logger.warning(f"Unsupported file type: {file_path.name}")
        return None, None

# ============================================================================
# MAIN INGESTION FUNCTION
# ============================================================================

def ingest_data(max_workers: int = 4) -> Dict[str, List]:
    """
    Ingest all data files with parallel processing
    
    Args:
        max_workers: Number of parallel workers (default: 4)
    
    Returns:
        dict: Categorized ingestion results
    """
    logger.info("=" * 70)
    logger.info("Starting Data Ingestion with Parallel Processing")
    logger.info("=" * 70)
    
    # Validate data folder
    if not DATA_FOLDER.exists():
        logger.error(f"Data folder not found: {DATA_FOLDER}")
        logger.info("Please create the Data folder and add your files")
        return {"structured": [], "semi_structured": [], "unstructured": []}
    
    # Collect all files
    all_files = [f for f in DATA_FOLDER.rglob("*") if f.is_file()]
    
    if not all_files:
        logger.warning(f"No files found in {DATA_FOLDER}")
        return {"structured": [], "semi_structured": [], "unstructured": []}
    
    logger.info(f"Found {len(all_files)} files to process")
    
    # Initialize results
    data = {
        "structured": [],
        "semi_structured": [],
        "unstructured": []
    }
    
    # Process files in parallel
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        futures = {executor.submit(process_single_file, f): f for f in all_files}
        
        # Collect results as they complete
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                category, result = future.result()
                
                if category and result:
                    data[category].append(result)
                    processed_count += 1
                    logger.info(f"✅ {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ {file_path.name}: {str(e)}")
    
    logger.info("=" * 70)
    logger.info(f"Ingestion Complete: {processed_count}/{len(all_files)} files processed")
    logger.info("=" * 70)
    
    return data

# ============================================================================
# SAVE AND SUMMARY
# ============================================================================

def save_results(data: Dict[str, List]):
    """Save ingestion results to JSON file"""
    try:
        with open(INGESTED_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved to: {INGESTED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_summary(data: Dict[str, List]):
    """Print ingestion summary"""
    total_rows = sum(item.get("rows", 0) for item in data["structured"])
    
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Structured files     : {len(data['structured'])}")
    print(f"Semi-structured files: {len(data['semi_structured'])}")
    print(f"Unstructured files   : {len(data['unstructured'])}")
    print(f"Total rows ingested  : {total_rows:,}")
    print("=" * 70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Run ingestion
        results = ingest_data(max_workers=4)
        
        # Save results
        save_results(results)
        
        # Print summary
        print_summary(results)
        
        logger.info("\n✅ Data ingestion complete!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()