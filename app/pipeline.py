"""
Automated PDF Processing Pipeline
1. Process all PDFs in 'pdfs' folder
2. Create CSV files in 'csv' folder using create_csv.py functions
3. Run predictions using trained model (enhanced_label_encoder.joblib)
4. Generate JSON outline files in 'output' directory
"""

import os
import pandas as pd
import joblib
import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from create_csv import (
        extract_pdf_features,
        filter_significant_elements,
        save_to_csv
    )
    CSV_MODULE_AVAILABLE = True
except ImportError as e:
    CSV_MODULE_AVAILABLE = False
    print(f"❌ Could not import create_csv.py: {e}")
    print("Make sure create_csv.py is in the same directory and properly named")


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['pdfs', 'csv',
                   'output']  # Changed 'results' to 'output'

    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"📁 Directory exists: {dir_name}")

    return directories


def load_trained_model():
    """Load the trained model and associated files"""
    try:
        # Load model components
        model = joblib.load('enhanced_pdf_heading_rf_model.joblib')
        label_encoder = joblib.load('enhanced_label_encoder.joblib')

        # Load metadata
        with open('enhanced_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, label_encoder, metadata

    except FileNotFoundError as e:
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None


def preprocess_csv_for_prediction(df, expected_features):
    """Preprocess CSV data to match training format - same as training preprocessing"""

    # Convert numeric columns
    numeric_columns = [
        'page_num', 'block_num', 'line_num', 'font_size', 'position_x', 'position_y',
        'line_width', 'line_height', 'page_width', 'page_height', 'char_count',
        'word_count', 'distance_from_left', 'distance_from_right',
        'distance_from_top', 'distance_from_bottom', 'space_above', 'space_below',
        'font_size_ratio'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                fill_value = df[col].median() if col != 'page_num' else 1
                df[col] = df[col].fillna(fill_value)

    # Create relative coordinates if missing
    if 'page_width' in df.columns and 'page_height' in df.columns:
        if 'relative_x' not in df.columns:
            df['position_x'] = pd.to_numeric(
                df['position_x'], errors='coerce').fillna(0)
            df['page_width'] = pd.to_numeric(
                df['page_width'], errors='coerce').fillna(1)
            df['page_width'] = df['page_width'].replace(0, 1)
            df['relative_x'] = df['position_x'] / df['page_width']
            df['relative_x'] = df['relative_x'].fillna(0)

        if 'relative_y' not in df.columns:
            df['position_y'] = pd.to_numeric(
                df['position_y'], errors='coerce').fillna(0)
            df['page_height'] = pd.to_numeric(
                df['page_height'], errors='coerce').fillna(1)
            df['page_height'] = df['page_height'].replace(0, 1)
            df['relative_y'] = df['position_y'] / df['page_height']
            df['relative_y'] = df['relative_y'].fillna(0)

    # Convert boolean columns (same as training)
    boolean_columns = [
        'is_bold', 'is_italic', 'is_superscript', 'is_all_caps',
        'is_title_case', 'has_numbers', 'starts_with_number',
        'is_largest_font', 'is_above_avg_font'
    ]

    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            boolean_map = {
                'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0,
                'YES': 1, 'NO': 0, 'T': 1, 'F': 0,
                '1.0': 1, '0.0': 0, 'NAN': 0, 'NONE': 0
            }
            df[col] = df[col].map(boolean_map)
            df[col] = pd.to_numeric(
                df[col], errors='coerce').fillna(0).astype(int)

    # Create composite features (same as training)
    formatting_features = ['is_bold', 'is_italic',
                           'is_all_caps', 'is_title_case']
    available_formatting = [
        col for col in formatting_features if col in df.columns]

    if available_formatting:
        df['formatting_score'] = df[available_formatting].sum(axis=1)

    if 'is_bold' in df.columns and 'font_size_ratio' in df.columns:
        df['is_bold'] = pd.to_numeric(df['is_bold'], errors='coerce').fillna(0)
        df['font_size_ratio'] = pd.to_numeric(
            df['font_size_ratio'], errors='coerce').fillna(1.0)
        df['font_emphasis'] = df['is_bold'] * df['font_size_ratio']

    # Position-based features
    if 'relative_y' in df.columns and 'relative_x' in df.columns:
        df['relative_y'] = pd.to_numeric(
            df['relative_y'], errors='coerce').fillna(0)
        df['relative_x'] = pd.to_numeric(
            df['relative_x'], errors='coerce').fillna(0)
        df['is_top_third'] = (df['relative_y'] < 0.33).astype(int)
        df['is_left_aligned'] = (df['relative_x'] < 0.1).astype(int)
    else:
        df['is_top_third'] = 0
        df['is_left_aligned'] = 0

    # Text length categories
    if 'char_count' in df.columns:
        df['char_count'] = pd.to_numeric(
            df['char_count'], errors='coerce').fillna(0)
        df['is_short_text'] = (df['char_count'] <= 5).astype(int)
        df['is_medium_text'] = ((df['char_count'] > 5) & (
            df['char_count'] <= 50)).astype(int)
        df['is_long_text'] = (df['char_count'] > 50).astype(int)
    else:
        df['is_short_text'] = 0
        df['is_medium_text'] = 1
        df['is_long_text'] = 0

    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in df.columns:
            print(f"   ⚠️  Missing feature '{feature}', setting to 0")
            df[feature] = 0

    # Fill any remaining missing values
    for col in expected_features:
        if df[col].isnull().sum() > 0:
            if col in boolean_columns + ['formatting_score']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"   ✅ Preprocessed shape: {df.shape}")
    return df


def create_outline_json(df, pdf_name):
    """Create JSON outline structure from predictions - Using model-predicted title"""

    # Filter predictions - separate title from other headings
    predictions_df = df[~df['predicted_label'].isin(
        ['body_text', 'body'])].copy()

    # Extract title from model predictions (first occurrence of 'title' prediction)
    title_rows = predictions_df[predictions_df['predicted_label'].str.lower(
    ) == 'title']

    if not title_rows.empty:
        # Use the first predicted title, sorted by page and position
        title_rows_sorted = title_rows.sort_values(['page_num', 'position_y'])
        document_title = title_rows_sorted.iloc[0]['text'].strip()
    else:
        # Fallback: if no title predicted, use filename
        document_title = pdf_name.replace('_', ' ').replace('-', ' ').title()
        print(
            f"   ⚠️  No title predicted by model, using filename: {document_title}")

    # Filter out title from outline (keep only H1, H2, H3, etc.)
    headings_df = predictions_df[~predictions_df['predicted_label'].str.lower().isin([
        'title'])].copy()

    # Sort headings by page number and position
    headings_df = headings_df.sort_values(['page_num', 'position_y'])

    # Create outline structure - only actual headings, not title
    outline = []

    for _, row in headings_df.iterrows():
        heading_item = {
            "level": row['predicted_label'].upper(),  # H1, H2, etc.
            "text": row['text'].strip(),
            "page": int(row['page_num'])
        }
        outline.append(heading_item)

    # Create the final JSON structure
    json_structure = {
        "title": document_title,  # Use model-predicted title
        "outline": outline  # Contains only H1, H2, H3, etc. - no title duplication
    }

    return json_structure


def process_single_pdf(pdf_path, csv_output_dir, model, label_encoder, expected_features):
    """Process a single PDF through the complete pipeline"""

    pdf_name = Path(pdf_path).stem

    try:
        # Step 1: Extract features using create_csv.py functions
        features = extract_pdf_features(pdf_path)

        if not features:
            return None

        # Step 2: Filter significant elements
        filtered_features, filter_stats = filter_significant_elements(features)

        if not filtered_features:
            print(f"   ⚠️  No significant elements after filtering, using original data")
            filtered_features = features

        # Step 3: Save CSV file in csv directory
        csv_filename = f"pdf_features_filtered-{pdf_name}.csv"
        csv_path = os.path.join(csv_output_dir, csv_filename)

        df = pd.DataFrame(filtered_features)
        df.to_csv(csv_path, index=False)

        # Step 4: Preprocess for prediction
        print("   🔧 Preprocessing for prediction...")
        df_processed = preprocess_csv_for_prediction(
            df.copy(), expected_features)

        # Step 5: Make predictions
        print("   🔮 Making predictions...")
        X = df_processed[expected_features].copy()

        predictions = model.predict(X)
        prediction_probs = model.predict_proba(X)
        predicted_labels = label_encoder.inverse_transform(predictions)
        confidence_scores = np.max(prediction_probs, axis=1)

        # Step 6: Create results dataframe
        results_df = df.copy()
        results_df['predicted_label'] = predicted_labels
        results_df['confidence'] = confidence_scores

        # Add probability columns
        for i, class_name in enumerate(label_encoder.classes_):
            results_df[f'prob_{class_name}'] = prediction_probs[:, i]

        # Step 7: Create JSON outline (FIXED - no duplicate title)
        json_outline = create_outline_json(results_df, pdf_name)

        # Save JSON outline
        json_filename = f"{pdf_name}.json"
        json_path = os.path.join("output", json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_outline, f, indent=4, ensure_ascii=False)

        # Step 8: Print summary
        pred_counts = pd.Series(predicted_labels).value_counts()
        print(f"   📈 Prediction Summary:")
        for label, count in pred_counts.items():
            percentage = (count / len(predicted_labels)) * 100
        return {
            'pdf_name': pdf_name,
            'csv_path': csv_path,
            'json_path': json_path,
            'total_elements': len(filtered_features),
            'total_headings': len(json_outline['outline']),
            'predictions': dict(pred_counts),
            'mean_confidence': confidence_scores.mean()
        }

    except Exception as e:
        print(f"   ❌ Error processing {pdf_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_pipeline():
    """Main pipeline function"""

    print("="*70)
    print("AUTOMATED PDF HEADING DETECTION PIPELINE")
    print("="*70)

    # Step 1: Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()

    # Step 2: Check if CSV creation module is available
    if not CSV_MODULE_AVAILABLE:
        print("\n❌ Cannot proceed without create_csv.py module")
        print("Please ensure create_csv.py is in the same directory")
        return

    # Step 3: Load trained model
    model, label_encoder, metadata = load_trained_model()
    if model is None:
        print("❌ Cannot proceed without trained model")
        return

    expected_features = metadata['feature_columns']

    # Step 4: Find PDFs to process
    pdf_dir = "pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        return

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf_file}")

    # Step 5: Process each PDF
    results = []
    successful = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, pdf_file)

        print(f"\n[{i}/{len(pdf_files)}] " + "="*50)

        result = process_single_pdf(
            pdf_path,
            "csv",
            model,
            label_encoder,
            expected_features
        )

        if result:
            results.append(result)
            successful += 1
            print(f"✅ Successfully processed: {pdf_file}")
        else:
            print(f"❌ Failed to process: {pdf_file}")

    # Step 6: Generate summary report

    if results:

        total_elements = sum(r['total_elements'] for r in results)
        total_headings = sum(r['total_headings'] for r in results)
        avg_confidence = sum(r['mean_confidence']
                             for r in results) / len(results)

        # Aggregate predictions
        all_predictions = {}
        for result in results:
            for label, count in result['predictions'].items():
                all_predictions[label] = all_predictions.get(label, 0) + count

        for label, count in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_elements) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")

        # Create summary report
        summary_data = []
        for result in results:
            summary_data.append({
                'pdf_name': result['pdf_name'],
                'total_elements': result['total_elements'],
                'total_headings': result['total_headings'],
                'mean_confidence': result['mean_confidence'],
                'json_file': f"{result['pdf_name']}.json"
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = "output/pipeline_summary.csv"
        summary_df.to_csv(summary_path, index=False)


def main():
    """Main entry point"""
    run_pipeline()


if __name__ == "__main__":
    main()
