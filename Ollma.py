
import requests
import json

print("="*70)
print("OLLAMA CONNECTION TEST")
print("="*70)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_LIST_URL = "http://localhost:11434/api/tags"

# Test 1: Check if Ollama is running
print("\n1. Testing Ollama connection...")
try:
    response = requests.get(OLLAMA_LIST_URL, timeout=5)
    if response.status_code == 200:
        print("✅ Ollama is running")
        
        # List available models
        models_data = response.json()
        models = models_data.get("models", [])
        
        if models:
            print(f"\n📦 Available models ({len(models)}):")
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                print(f"  • {name} ({size:.1f} GB)")
        else:
            print("\n⚠️  No models found. Download one with:")
            print("   ollama pull llama3.2")
    else:
        print(f"❌ Ollama returned status {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Ollama")
    print("\n💡 Solutions:")
    print("   1. Install Ollama from: https://ollama.com/download")
    print("   2. Start Ollama server: ollama serve")
    print("   3. Wait a few seconds and try again")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 2: Test model inference
if models:
    print("\n2. Testing model inference...")
    
    # Use first available model
    test_model = models[0].get("name")
    print(f"   Using model: {test_model}")
    
    test_prompt = "What is 2+2? Answer in one short sentence."
    
    try:
        print("   Sending test query...")
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": test_model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            
            print("✅ Model inference successful")
            print(f"\n   Question: {test_prompt}")
            print(f"   Answer: {answer}")
            
            # Check response quality
            if "4" in answer:
                print("\n🎉 Ollama is working correctly!")
            else:
                print("\n⚠️  Model responded but answer seems incorrect")
                print("   This might be normal for some models")
        else:
            print(f"❌ API returned status {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model might be loading)")
        print("   Try running the test again in a few seconds")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

# Test 3: Check recommended models
print("\n3. Checking recommended models...")

recommended_models = {
    "llama3.2": "Best overall quality",
    "mistral": "Great for reasoning",
    "phi": "Fastest, lightweight"
}

available_model_names = [m.get("name", "").split(":")[0] for m in models]

for model, description in recommended_models.items():
    if model in available_model_names:
        print(f"   ✅ {model:15} - {description} (installed)")
    else:
        print(f"   ⬜ {model:15} - {description} (not installed)")
        print(f"      Download: ollama pull {model}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if models:
    print("\n✅ Ollama is ready to use!")
    print("\nNext steps:")
    print("   1. Configure model in rag_backend_ollama.py:")
    print(f"      OLLAMA_MODEL = \"{test_model.split(':')[0]}\"")
    print("   2. Start your RAG backend:")
    print("      python rag_backend_ollama.py")
else:
    print("\n⚠️  Ollama is running but no models are installed")
    print("\nNext steps:")
    print("   1. Download a model:")
    print("      ollama pull llama3.2")
    print("   2. Run this test again:")
    print("      python test_ollama.py")

print("\n" + "="*70)

# Test 4: Performance benchmark (optional)
if models:
    print("\n4. Optional: Run performance benchmark? (y/N): ", end="")
    try:
        choice = input().lower()
        if choice == 'y':
            print("\n   Running benchmark...")
            
            benchmark_prompts = [
                "What is AI?",
                "Who invented the telephone?",
                "Explain machine learning in one sentence."
            ]
            
            import time
            
            for i, prompt in enumerate(benchmark_prompts, 1):
                print(f"\n   Test {i}/3: {prompt}")
                
                start = time.time()
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": test_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 100}
                    },
                    timeout=30
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    print(f"   ⏱️  Response time: {elapsed:.2f}s")
                    print(f"   📝 Answer: {answer[:100]}...")
                else:
                    print(f"   ❌ Failed (status {response.status_code})")
            
            print("\n   ✅ Benchmark complete")
    except KeyboardInterrupt:
        print("\n   Skipped.")
    except:
        pass

print("\nFor detailed setup instructions, see: OLLAMA_SETUP_GUIDE.md")