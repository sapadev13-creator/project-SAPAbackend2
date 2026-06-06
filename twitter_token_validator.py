#!/usr/bin/env python3
"""
Twitter Bearer Token Decoder & Validator
Membantu debug dan verify Twitter API v2 credentials
"""

import urllib.parse
import sys

def decode_bearer_token(token):
    """Decode URL-encoded bearer token"""
    print("\n" + "="*70)
    print("🔍 BEARER TOKEN ANALYSIS")
    print("="*70)
    
    print(f"\n📝 Original Token (first 50 chars):")
    print(f"   {token[:50]}...")
    
    # Check if URL-encoded
    has_encoding = '%' in token
    print(f"\n🔍 URL Encoding Check:")
    print(f"   {'❌ URL-ENCODED (HAS %)' if has_encoding else '✅ NOT URL-ENCODED'}")
    
    if has_encoding:
        print(f"   Found encoded characters: ", end="")
        encoded_chars = [char for char in token if char == '%']
        print(f"{len(encoded_chars)} instances")
    
    # Try to decode if encoded
    if has_encoding:
        try:
            decoded = urllib.parse.unquote(token)
            print(f"\n📝 Decoded Token (first 50 chars):")
            print(f"   {decoded[:50]}...")
            return decoded
        except Exception as e:
            print(f"\n❌ Error decoding: {str(e)}")
            return token
    
    return token

def validate_bearer_token(token):
    """Validate bearer token format"""
    print("\n" + "="*70)
    print("✅ BEARER TOKEN VALIDATION")
    print("="*70)
    
    checks = {
        "Starts with 'AAAAAA'": token.startswith("AAAAAA"),
        "Length > 100": len(token) > 100,
        "Contains only valid chars": all(c.isalnum() or c in "=-_" for c in token),
        "No spaces": " " not in token,
        "No URL encoding": "%" not in token,
    }
    
    print("\nValidation Results:")
    all_passed = True
    for check, result in checks.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {check}: {'PASS' if result else 'FAIL'}")
        if not result:
            all_passed = False
    
    return all_passed

def check_token_format(token):
    """Check and display token format details"""
    print("\n" + "="*70)
    print("📊 TOKEN FORMAT DETAILS")
    print("="*70)
    
    print(f"\nToken Length: {len(token)} characters")
    print(f"Token Type: Bearer Token v2")
    print(f"\nFirst 20 chars: {token[:20]}")
    print(f"Last 20 chars:  {token[-20:]}")
    
    # Count character types
    alphanumeric = sum(1 for c in token if c.isalnum())
    special = sum(1 for c in token if not c.isalnum())
    
    print(f"\nCharacter breakdown:")
    print(f"  Alphanumeric: {alphanumeric}")
    print(f"  Special chars: {special}")
    
    # Check for common issues
    print(f"\nPotential Issues:")
    issues = []
    
    if "=" in token:
        issues.append("  ⚠️  Contains '=' - might be base64 encoded")
    if "%3D" in token:
        issues.append("  ❌ Contains '%3D' - URL-ENCODED EQUALS SIGN")
    if "%2F" in token:
        issues.append("  ❌ Contains '%2F' - URL-ENCODED SLASH")
    if "%20" in token:
        issues.append("  ❌ Contains '%20' - URL-ENCODED SPACE")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✅ No obvious encoding issues detected")

def generate_env_line(token, decoded_token):
    """Generate correct .env line"""
    print("\n" + "="*70)
    print("📝 CORRECT .ENV FORMAT")
    print("="*70)
    
    print(f"\nAdd this line to your .env file:")
    print(f"\nTWITTER_BEARER_TOKEN={decoded_token}")
    
    print(f"\n✅ This token is ready to use!")
    print(f"\nDO NOT include this in version control:")
    print(f"  - Add .env to .gitignore")
    print(f"  - Never commit secrets")

def main():
    if len(sys.argv) < 2:
        print("\n🔧 Twitter Bearer Token Validator")
        print("="*70)
        print("\nUsage:")
        print("  python twitter_token_validator.py <bearer_token>")
        print("\nExample:")
        print("  python twitter_token_validator.py 'AAAAAAAAAAAAAAAAAAAAAInq6wEAAAAAcCNTVdT3cqOTQH8cn0XE8vbOapg%3DcmUWPZQSLdwGH3JRFqbx3pmpVkojuXdom5syjPnT0LqYvLXsZc'")
        print("\nThe script will:")
        print("  1. Detect if token is URL-encoded")
        print("  2. Decode it if necessary")
        print("  3. Validate token format")
        print("  4. Show correct .env format")
        print("="*70 + "\n")
        sys.exit(1)
    
    token = sys.argv[1]
    
    # Decode if needed
    decoded_token = decode_bearer_token(token)
    
    # Validate format
    is_valid = validate_bearer_token(decoded_token)
    
    # Show format details
    check_token_format(decoded_token)
    
    # Generate .env line
    generate_env_line(token, decoded_token)
    
    # Final status
    print("\n" + "="*70)
    if is_valid:
        print("✅ TOKEN IS VALID AND READY TO USE!")
    else:
        print("❌ TOKEN HAS ISSUES - CHECK ABOVE FOR DETAILS")
    print("="*70 + "\n")
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())
