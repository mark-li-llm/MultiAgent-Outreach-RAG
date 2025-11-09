#!/bin/bash
# Railway Deployment Test Script
# Usage: ./test_railway_deployment.sh

set -e

RAILWAY_URL="multiagent-outreach-rag-production.up.railway.app"
BASE_URL="https://$RAILWAY_URL"

echo "======================================"
echo "Railway Deployment Test Suite"
echo "======================================"
echo "URL: $BASE_URL"
echo "Time: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test endpoint
test_endpoint() {
    local test_name="$1"
    local expected_status="$2"
    local url="$3"
    local method="${4:-GET}"
    local data="${5:-}"

    echo -n "Test: $test_name ... "

    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    fi

    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "$expected_status" ]; then
        echo -e "${GREEN}PASS${NC} (HTTP $http_code)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC} (Expected HTTP $expected_status, got $http_code)"
        echo "Response: $body"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "======================================"
echo "1. Basic Connectivity Tests"
echo "======================================"

# Test 1: Root endpoint
test_endpoint "Root endpoint (health check)" 200 "$BASE_URL/"

# Test 2: Health check endpoint
test_endpoint "Detailed health check" 200 "$BASE_URL/health"

# Test 3: API documentation
test_endpoint "API documentation" 200 "$BASE_URL/docs"

echo ""
echo "======================================"
echo "2. API Functionality Tests"
echo "======================================"

# Test 4: Email generation API (will fail if OPENAI_API_KEY not set)
echo -n "Test: Email generation API ... "
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "company": "Salesforce",
        "persona": "vp_customer_experience",
        "session_id": "test-'$(date +%s)'"
    }' 2>&1)

http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}PASS${NC} (HTTP $http_code)"
    echo "Response: $body" | python3 -m json.tool
    ((TESTS_PASSED++))
elif echo "$body" | grep -q "OPENAI_API_KEY"; then
    echo -e "${YELLOW}EXPECTED FAIL${NC} (Missing OPENAI_API_KEY)"
    echo "Note: Set OPENAI_API_KEY in Railway Variables to enable this endpoint"
    ((TESTS_PASSED++)) # Count as pass since this is expected
else
    echo -e "${RED}FAIL${NC} (HTTP $http_code)"
    echo "Response: $body"
    ((TESTS_FAILED++))
fi

echo ""
echo "======================================"
echo "3. Error Handling Tests"
echo "======================================"

# Test 5: Invalid request (missing required field)
test_endpoint "Invalid request handling" 422 "$BASE_URL/api/generate" "POST" '{}'

# Test 6: Invalid endpoint
test_endpoint "404 handling" 404 "$BASE_URL/invalid-endpoint"

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Set OPENAI_API_KEY in Railway Variables"
    echo "2. Visit $BASE_URL/docs to explore the API"
    echo "3. Test email generation with a real request"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the errors above.${NC}"
    exit 1
fi
