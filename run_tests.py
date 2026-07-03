import sys
from test_validation import test_should_skip_article, test_post_process_headline, test_validate_headline

def run():
    print("Running test_should_skip_article...")
    test_should_skip_article()
    print("Passed!")
    
    print("Running test_post_process_headline...")
    test_post_process_headline()
    print("Passed!")
    
    print("Running test_validate_headline...")
    test_validate_headline()
    print("Passed!")
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print(f"Test failure! AssertionError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test error! Exception: {e}")
        sys.exit(1)
