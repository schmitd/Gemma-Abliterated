#!/usr/bin/env python3
"""
Test script to verify all modules can be imported correctly.
"""

def test_imports():
    """Test importing all modules."""
    
    print("Testing module imports...")
    
    try:
        from src.config import MODEL_ID, GEMMA3_CHAT_TEMPLATE
        print("✓ config.py imported successfully")
        
        from src.utils import setup_logging, get_token_id_set
        print("✓ utils.py imported successfully")
        
        from src.model_manager import ModelManager
        print("✓ model_manager.py imported successfully")
        
        from src.orthogonalization import apply_weight_orthogonalization
        print("✓ orthogonalization.py imported successfully")
        
        from src.evaluation import benchmark_model, compare_results
        print("✓ evaluation.py imported successfully")
        
        from src.model_saver import ModelSaver
        print("✓ model_saver.py imported successfully")
        
        print("\nAll modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration values."""
    
    print("\nTesting configuration...")
    
    from src.config import MODEL_ID, DEFAULT_SAMPLED_TOKENS, POSITIVE_STRINGS
    
    print(f"Model ID: {MODEL_ID}")
    print(f"Default sampled tokens: {DEFAULT_SAMPLED_TOKENS}")
    print(f"Number of positive strings: {len(POSITIVE_STRINGS)}")
    
    print("✓ Configuration test passed")

def test_utils():
    """Test utility functions."""
    
    print("\nTesting utility functions...")
    
    from src.utils import setup_logging, create_refusal_regex
    
    # Test logging setup
    setup_logging()
    print("✓ Logging setup successful")
    
    # Test regex creation
    regex = create_refusal_regex()
    print(f"✓ Refusal regex created: {type(regex)}")
    
    print("✓ Utility functions test passed")

if __name__ == "__main__":
    print("="*50)
    print("MODULE TESTING")
    print("="*50)
    
    success = test_imports()
    
    if success:
        test_config()
        test_utils()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
        print("The modular structure is working correctly.")
    else:
        print("\n" + "="*50)
        print("TESTS FAILED!")
        print("="*50)
        print("Please check the import errors above.")
