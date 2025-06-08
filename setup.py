import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama is not installed or not in PATH")
        return False

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print("Installing Ollama...")
    
    if system == "linux":
        print("For Linux, run: curl -fsSL https://ollama.ai/install.sh | sh")
    elif system == "darwin":  # macOS
        print("For macOS, download from: https://ollama.ai/download/mac")
    elif system == "windows":
        print("For Windows, download from: https://ollama.ai/download/windows")
    else:
        print(f"Please install Ollama manually for {system}")
    
    print("After installing Ollama, run this script again.")
    return False

def download_llm_model(model_name="llama3.2"):
    """Download the specified LLM model"""
    try:
        print(f"📥 Downloading {model_name} model...")
        print("This may take several minutes depending on your internet connection...")
        
        result = subprocess.run(['ollama', 'pull', model_name], 
                              check=True, capture_output=False)
        print(f"✅ {model_name} model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {model_name} model: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'Text_v1.txt',
        'Text_v2.txt',
        'requirements.txt',
        'document_processor.py',
        'chatbot_engine.py',
        'streamlit_app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        
        # Provide activation instructions
        system = platform.system().lower()
        if system == "windows":
            activate_cmd = "venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print(f"To activate the virtual environment, run: {activate_cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def process_documents():
    """Process the documents and create vector store"""
    try:
        print("📄 Processing documents...")
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        document_paths = ["Text_v1.txt", "Text_v2.txt"]
        
        success = processor.process_documents(document_paths)
        if success:
            print("✅ Documents processed successfully")
            return True
        else:
            print("❌ Failed to process documents")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 GCP Regulatory Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check required files
    if not check_required_files():
        print("\n❌ Setup failed: Missing required files")
        sys.exit(1)
    
    # Check Ollama installation
    if not check_ollama_installation():
        if not install_ollama():
            print("\n❌ Setup failed: Ollama installation required")
            sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Setup failed: Could not install Python dependencies")
        sys.exit(1)
    
    # Download LLM model
    print("\n🤖 Setting up LLM model...")
    model_choice = input("Choose LLM model (1: llama3.2, 2: mistral, 3: phi3): ").strip()
    
    model_map = {
        "1": "llama3.2",
        "2": "mistral",
        "3": "phi3"
    }
    
    model_name = model_map.get(model_choice, "llama3.2")
    
    if not download_llm_model(model_name):
        print(f"\n⚠️ Warning: Failed to download {model_name} model")
        print("You can download it later using: ollama pull <model_name>")
    
    # Process documents
    print("\n📄 Processing documents...")
    if process_documents():
        print("✅ Document processing completed")
    else:
        print("⚠️ Warning: Document processing failed")
        print("You can process documents later by running the app")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run streamlit_app.py")
    print("\nOr use the run script:")
    print("python run_app.py")

if __name__ == "__main__":
    main()