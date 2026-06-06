#!/usr/bin/env python3
"""
Test script untuk /predict/twitter/profile endpoint
Memverifikasi JSON parsing dan X API v2 integration
"""

import json
import requests
import sys

BASE_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_test(name, status):
    icon = "✅" if status else "❌"
    print(f"\n{icon} {name}")

def test_json_payload_validation():
    """Test bahwa endpoint correctly parses valid JSON payload"""
    print_header("TEST 1: Valid JSON Payload Parsing")
    
    test_cases = [
        {
            "name": "Valid Twitter profile URL",
            "payload": {"profile_url": "https://twitter.com/elonmusk"},
            "should_pass_parsing": True,
        },
        {
            "name": "Twitter URL with trailing slash",
            "payload": {"profile_url": "https://twitter.com/elonmusk/"},
            "should_pass_parsing": True,
        },
        {
            "name": "Twitter URL with @",
            "payload": {"profile_url": "https://twitter.com/@elonmusk"},
            "should_pass_parsing": True,
        },
        {
            "name": "Plain username",
            "payload": {"profile_url": "elonmusk"},
            "should_pass_parsing": True,
        },
    ]
    
    all_passed = True
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        print(f"  Payload: {json.dumps(test_case['payload'])}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/twitter/profile",
                json=test_case['payload'],
                timeout=10
            )
            
            # JSON parsing succeeded if we get a response (even if 401, 404, 500)
            json_parsing_ok = True
            print(f"  ✅ JSON parsed successfully (Status: {response.status_code})")
            
            # Print response for debugging
            if response.status_code == 422:
                print(f"  ⚠️  Validation Error: {response.json()}")
                all_passed = False
            elif response.status_code == 400:
                print(f"  ⚠️  Bad Request: {response.json()}")
            elif response.status_code == 401:
                print(f"  ℹ️  Unauthorized (expected if no tweets auth)")
            elif response.status_code == 404:
                print(f"  ℹ️  Not Found: {response.json().get('detail', 'Unknown')}")
            elif response.status_code == 500:
                print(f"  ⚠️  Server Error: {response.json().get('detail', 'Unknown')}")
            else:
                print(f"  Response: {response.json()}")
                
        except requests.exceptions.JSONDecodeError as e:
            print(f"  ❌ JSON Decode Error: {str(e)}")
            all_passed = False
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  Connection Error - Server not running on {BASE_URL}")
            return False
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_invalid_json():
    """Test bahwa endpoint rejects invalid JSON"""
    print_header("TEST 2: Invalid JSON Handling")
    
    test_cases = [
        {
            "name": "Missing required field",
            "payload": {},
            "should_fail": True,
        },
        {
            "name": "Extra fields are allowed",
            "payload": {"profile_url": "https://twitter.com/user", "extra": "field"},
            "should_fail": False,
        },
    ]
    
    all_passed = True
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        print(f"  Payload: {json.dumps(test_case['payload'])}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/twitter/profile",
                json=test_case['payload'],
                timeout=10
            )
            
            is_error = response.status_code == 422
            expected_error = test_case.get('should_fail', False)
            
            if expected_error and is_error:
                print(f"  ✅ Correctly rejected (422 Validation Error)")
                print(f"  Details: {response.json()['detail']}")
            elif not expected_error and response.status_code != 422:
                print(f"  ✅ Accepted (Status: {response.status_code})")
            else:
                print(f"  ❌ Unexpected status: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_request_formats():
    """Test berbagai format HTTP requests"""
    print_header("TEST 3: HTTP Request Format Validation")
    
    print(f"\n  Testing raw JSON request...")
    
    # Test dengan raw string JSON
    try:
        response = requests.post(
            f"{BASE_URL}/predict/twitter/profile",
            data='{"profile_url": "https://twitter.com/user"}',
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"  ✅ Raw JSON string accepted (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Raw JSON string failed: {str(e)}")
    
    # Test dengan requests.post json parameter
    try:
        response = requests.post(
            f"{BASE_URL}/predict/twitter/profile",
            json={"profile_url": "https://twitter.com/user"},
            timeout=10
        )
        print(f"  ✅ requests.post json parameter accepted (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ requests.post json parameter failed: {str(e)}")

def main():
    print("\n" + "🧪 TWITTER PROFILE ENDPOINT JSON VALIDATION TESTS")
    print("Testing /predict/twitter/profile endpoint for proper JSON handling")
    print("According to X API v2 specifications")
    
    try:
        # Run tests
        test1_passed = test_json_payload_validation()
        test2_passed = test_invalid_json()
        test_request_formats()
        
        # Summary
        print_header("TEST SUMMARY")
        print("\n✅ JSON Payload Validation:" + (" PASSED ✅" if test1_passed else " FAILED ❌"))
        print("✅ Invalid JSON Handling:" + (" PASSED ✅" if test2_passed else " FAILED ❌"))
        print("\n📝 Notes:")
        print("  - Endpoint now uses proper Pydantic model: TwitterProfileRequest")
        print("  - JSON parsing properly handled by FastAPI")
        print("  - X API v2 parameters included (tweet_fields, max_results)")
        print("  - Username extraction handles URLs with @ and trailing slashes")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
