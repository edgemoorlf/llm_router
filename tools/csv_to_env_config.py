import sys
import argparse
import pandas as pd

def convert_csv_to_env(csv_path, output_path=None):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        output = []
        instance_ids = []
        
        for _, row in df.iterrows():
            if pd.isna(row['ID']) or pd.isna(row['API_KEY']):
                continue
                
            original_id = str(row['ID']).strip()
            instance_id = original_id.upper()
            instance_ids.append(original_id.lower())  # Ensure lowercase for API_INSTANCES line
            
            # API_KEY
            if pd.notna(row['API_KEY']):
                output.append(f"API_INSTANCE_{instance_id}_API_KEY={str(row['API_KEY']).strip()}")
            
            # API_BASE
            if pd.notna(row['API_BASE']):
                output.append(f"API_INSTANCE_{instance_id}_API_BASE={str(row['API_BASE']).strip()}")
            
            # PROXY_URL
            if pd.notna(row['PROXY_URL']):
                output.append(f"API_INSTANCE_{instance_id}_PROXY_URL={str(row['PROXY_URL']).strip()}")

            # Additional configuration
            output.append(f"API_INSTANCE_{instance_id}_API_VERSION=2024-08-01-preview")
            output.append(f"API_INSTANCE_{instance_id}_PRIORITY=1")
            output.append(f"API_INSTANCE_{instance_id}_WEIGHT=100")
            output.append(f"API_INSTANCE_{instance_id}_MAX_TPM=30000")
            
            # Get model info from CSV columns
            model_name = str(row['模型名字']).strip() if pd.notna(row['模型名字']) else ''
            model_version = str(row['模型版本']).strip() if pd.notna(row['模型版本']) else ''
            
            # Handle model name/version combination explicitly
            if model_name and model_version:
                supported_models = f"{model_name},{model_version}"
            elif model_name:
                supported_models = model_name
            elif model_version:
                supported_models = model_version
            else:
                supported_models = ''
            output.append(f"API_INSTANCE_{instance_id}_SUPPORTED_MODELS={supported_models}")
            
            output.append(f"API_INSTANCE_{instance_id}_MODEL_MAP_GPT4O_2024_11_20=gpt-4o")
            output.append(f"API_INSTANCE_{instance_id}_MODEL_MAP_GPT35TURBO=gpt-35-turbo")
            output.append(f"API_INSTANCE_{instance_id}_MAX_INPUT_TOKENS=4096")
            output.append("")  # Add empty line between instances

        # Add API_INSTANCES line at the top
        if instance_ids:
            output.insert(0, f"API_INSTANCES={','.join(instance_ids)}\n")
        
        # Remove trailing empty line from last instance
        if output and output[-1] == "":
            output.pop()
        
        # Add global model mapping
        output.append("MODEL_MAP_GPT4O=gpt-4o")
        result = '\n'.join(output).strip()
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        return result
                    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV API configurations to environment format')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    args = parser.parse_args()
    
    config_output = convert_csv_to_env(args.csv_path, args.output)
    print(config_output)
