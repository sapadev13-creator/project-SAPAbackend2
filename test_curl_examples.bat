@echo off
REM Test cases untuk /predict/twitter/profile endpoint di Windows

setlocal enabledelayedexpansion
set BASE_URL=http://localhost:8000

echo.
echo ===================================================================
echo Twitter Profile Detection Endpoint - Test Cases (Windows)
echo ===================================================================

echo.
echo TEST 1: Valid Twitter Profile URL
echo Command:
echo curl -X POST "%BASE_URL%/predict/twitter/profile" ^
echo   -H "Content-Type: application/json" ^
echo   -d "{\"profile_url\": \"https://twitter.com/elonmusk\"}"
echo.
echo Expected Response:
echo - Status 200 with OCEAN personality scores (if tweets found)
echo - Status 404 if user/tweets not found
echo - Status 500 if API error
echo.

echo TEST 2: Invalid JSON (Missing profile_url)
echo Command:
echo curl -X POST "%BASE_URL%/predict/twitter/profile" ^
echo   -H "Content-Type: application/json" ^
echo   -d "{}"
echo.
echo Expected Response:
echo - Status 422 Validation Error
echo - Error message: 'Field required'
echo.

echo TEST 3: Malformed JSON
echo Command:
echo curl -X POST "%BASE_URL%/predict/twitter/profile" ^
echo   -H "Content-Type: application/json" ^
echo   -d "{profile_url: \"https://twitter.com/user\"}"
echo.
echo Expected Response:
echo - Before fix: Status 422 JSON decode error
echo - After fix: Should be handled properly by Pydantic
echo.

echo TEST 4: Twitter URL with trailing slash
echo Command:
echo curl -X POST "%BASE_URL%/predict/twitter/profile" ^
echo   -H "Content-Type: application/json" ^
echo   -d "{\"profile_url\": \"https://twitter.com/user/\"}"
echo.
echo Expected Response:
echo - Same as TEST 1 (username extracted correctly)
echo.

echo TEST 5: Plain username
echo Command:
echo curl -X POST "%BASE_URL%/predict/twitter/profile" ^
echo   -H "Content-Type: application/json" ^
echo   -d "{\"profile_url\": \"username\"}"
echo.
echo Expected Response:
echo - Same as TEST 1 (username used as-is)
echo.

echo ===================================================================
echo QUICK TEST - Copy and paste one of these:
echo ===================================================================
echo.
echo curl -X POST "%BASE_URL%/predict/twitter/profile" -H "Content-Type: application/json" -d "{\"profile_url\": \"https://twitter.com/elonmusk\"}"
echo.
echo ===================================================================
echo Notes:
echo - Endpoint now has proper Pydantic validation
echo - JSON decode error should be FIXED
echo - Make sure TWITTER_BEARER_TOKEN is set in .env
echo - For local testing: python -m uvicorn sapa_api.app_factory:app --reload
echo ===================================================================
echo.
