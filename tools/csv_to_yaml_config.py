import sys
import argparse
import os
import pandas as pd
import yaml

def generate_deployment_name(model_name):
    """
    Generate an Azure deployment name from a model name, following Azure naming conventions.
    
    Args:
        model_name: Original model name (e.g., gpt-4o)
        
    Returns:
        Deployment name suitable for Azure (e.g., gpt4o)
    """
    # Remove hyphens and periods, lowercase
    deployment_name = model_name.replace('-', '').replace('.', '').lower()
    
    # Handle common OpenAI model names
    if model_name.startswith('gpt-4'):
        if 'o' in model_name:
            return 'gpt-4o'
        elif 'turbo' in model_name:
            return 'gpt4turbo'
        else:
            return 'gpt4'
    elif model_name.startswith('gpt-3.5'):
        return 'gpt35turbo'
    
    # Fallback to the general transformation
    return deployment_name

def convert_csv_to_yaml(csv_path, output_path=None, preserve_base=True, api_version="2024-08-01-preview"):
    """
    Convert CSV configuration file to YAML format for Azure OpenAI instances.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to write output YAML file
        preserve_base: Whether to include base configuration in output (True) 
                       or just return instances array (False)
        api_version: Azure OpenAI API version to use for all instances
        
    Returns:
        YAML string representation of the configuration
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Initialize config structure based on preserve_base flag
        if preserve_base:
            # Full configuration with base settings
            config = {
                "name": "Azure OpenAI Proxy",
                "version": "1.5.0",
                "port": 3010,
                "routing": {
                    "strategy": "round_robin",
                    "retries": 3,
                    "timeout": 60
                },
                "logging": {
                    "level": "INFO",
                    "file": "../logs/app.log",
                    "max_size": 5242880,
                    "backup_count": 3,
                    "feishu_webhook": "${FEISHU_WEBHOOK_URL}"
                },
                "monitoring": {
                    "stats_window_minutes": 5,
                    "additional_windows": [15, 30, 60]
                },
                "instances": []
            }
        else:
            # Only instances array
            config = {"instances": []}
        
        # Process each row in the CSV
        for _, row in df.iterrows():
            if pd.isna(row['ID']) or pd.isna(row['API_KEY1']):
                print(f"# Skipping row with missing ID or API_KEY: {row.to_dict()}")
                continue
            
            if row['STATUS'] in ['401', '异常']:
                print(f"# Skipping row abnormal status: {row.to_dict()}")
                continue

            instance_id = str(row['ID']).strip().lower()
            
            # Create instance configuration
            instance_config = {
                "name": instance_id,
                "provider_type": "azure",  # Default to Azure provider
                "api_key": str(row['API_KEY1']).strip(),
                "api_version": api_version,
                "priority": 100,
                "weight": 100,
                "max_tpm": 30000,
                "max_input_tokens": 8000
            }
            
            # Add API base if available
            if pd.notna(row['API_BASE']):
                instance_config["api_base"] = str(row['API_BASE']).strip()
                
            # Add proxy URL if available
            if pd.notna(row['PROXY_URL']):
                instance_config["proxy_url"] = str(row['PROXY_URL']).strip()
            
            # Handle supported models
            model_name = str(row['模型名字']).strip() if pd.notna(row['模型名字']) else ''
            model_version = str(row['模型版本']).strip() if pd.notna(row['模型版本']) else ''
            
            supported_models = []
            model_deployments = {}
            
            if model_name:
                supported_models.append(model_name)
                # Generate appropriate deployment name
                model_deployments[model_name] = generate_deployment_name(model_name)
                
            if model_version:
                supported_models.append(model_version)
                model_deployments[model_version] = generate_deployment_name(model_version)
                
            if supported_models:
                instance_config["supported_models"] = supported_models
                instance_config["model_deployments"] = model_deployments
            
            # Add instance to configuration array
            config["instances"].append(instance_config)
            
        # Log summary
        print(f"# Processed {len(config['instances'])} valid instances from CSV")
        
        # Convert to YAML
        yaml_output = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_output)
                
        return yaml_output
                    
    except Exception as e:
        print(f"# Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV API configurations to YAML format')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--instances-only', action='store_true', 
                        help='Output only the instances array without base configuration')
    parser.add_argument('--api-version', default='2024-08-01-preview',
                        help='Azure OpenAI API version to use (default: 2024-08-01-preview)')
    args = parser.parse_args()
    
    # Set preserve_base to False if instances-only flag is used
    preserve_base = not args.instances_only
    
    yaml_output = convert_csv_to_yaml(
        args.csv_path, 
        args.output, 
        preserve_base=preserve_base,
        api_version=args.api_version
    )
    
    if not args.output:
        print(yaml_output)
    else:
        print(f"# Configuration written to {args.output}")
