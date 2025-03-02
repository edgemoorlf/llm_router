#!/bin/bash
# Setup script for Azure OpenAI Proxy

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Azure OpenAI Proxy...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Copy .env.example to .env if .env doesn't exist
if [ ! -f "app/.env" ]; then
    echo -e "${GREEN}Creating .env file from example...${NC}"
    cp app/.env.example app/.env
    echo -e "${YELLOW}Please update app/.env with your Azure OpenAI credentials.${NC}"
else
    echo -e "${YELLOW}app/.env already exists. Skipping creation.${NC}"
fi

# Install optional packages for testing
echo -e "${GREEN}Installing optional packages for testing...${NC}"
pip install openai

# Make run.py executable
chmod +x run.py

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To start the proxy server:${NC}"
echo -e "    source venv/bin/activate"
echo -e "    ./run.py"
echo -e "${YELLOW}Or:${NC}"
echo -e "    source venv/bin/activate"
echo -e "    python run.py"
echo -e ""
echo -e "${YELLOW}To test the proxy:${NC}"
echo -e "    source venv/bin/activate"
echo -e "    python tests/test_proxy.py"
