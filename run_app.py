import os
import sys
import subprocess
import time
from pathlib import Path

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

def start_ollama():
    """Start Ollama service"""
    try:
        print("🚀 Starting Ollama service...")
        
        # Try to start Ollama (this may vary by system)
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait a moment for service to start
        time.sleep(3)
        
        if check_ollama_running():
            print("✅ Ollama service started successfully")
            return True
        else:
            print("⚠️ Ollama may not have started properly")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'streamlit',
        'langchain',
        'chromadb',
        'sentence_transformers'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing dependencies: {missing_modules}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_models():
    """Check available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, check=True)
        
        models = result.stdout.strip().split('\n')[1:]  # Skip header
        if models and models[0]:  # Check if any models exist
            print("✅ Available models:")
            for model in models:
                if model.strip():
                    print(f"   - {model.split()[0]}")
            return True
        else:
            print("⚠️ No models found")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ Could not list models")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        print("🎉 Starting GCP Regulatory Chatbot...")
        print("📱 The app will open in your web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n⚠️ To stop the application, press Ctrl+C in this terminal")
        print("=" * 60)
        
        # Run Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_app.py',
            '--server.port=8501',
            '--server.headless=false',
            '--browser.gatherUsageStats=false'
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return True

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'streamlit_app.py',
        'document_processor.py',
        'chatbot_engine.py',
        'Text_v1.txt',
        'Text_v2.txt'
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

def main():
    """Main function to run the application"""
    print("🏥 GCP Regulatory Chatbot")
    print("=" * 40)
    
    # Check required files
    if not check_required_files():
        print("\n❌ Cannot start: Missing required files")
        print("Please ensure all files are in the correct directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start: Missing dependencies")
        print("Please run the setup script first: python setup.py")
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama_running():
        print("⚠️ Ollama service is not running")
        if not start_ollama():
            print("❌ Cannot start Ollama service")
            print("Please start Ollama manually:")
            print("  ollama serve")
            sys.exit(1)
    else:
        print("✅ Ollama service is running")
    
    # Check models
    if not check_models():
        print("⚠️ No models available")
        print("Please download a model first:")
        print("  ollama pull llama3.2")
        # Don't exit here, let the app handle it
    
    print("\n🔧 All checks passed!")
    print("🚀 Starting application...\n")
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()