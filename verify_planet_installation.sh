#!/bin/bash
# Quick verification script to check if PlaNet integration is working
# This does NOT require CARLA or training, just checks the code

echo "=============================================="
echo "PlaNet Integration Verification"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
WARNINGS=0
FAILURES=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    SUCCESS=$((SUCCESS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILURES=$((FAILURES + 1))
}

echo "1. Checking file structure..."
echo ""

# Check if key files exist
if [ -f "dreamerv3/nets.py" ]; then
    check_pass "dreamerv3/nets.py exists"
else
    check_fail "dreamerv3/nets.py not found"
fi

if [ -f "dreamerv3/agent.py" ]; then
    check_pass "dreamerv3/agent.py exists"
else
    check_fail "dreamerv3/agent.py not found"
fi

if [ -f "dreamerv3/dreamerv3.yaml" ]; then
    check_pass "dreamerv3/dreamerv3.yaml exists"
else
    check_fail "dreamerv3/dreamerv3.yaml not found"
fi

if [ -f "PLANET_INTEGRATION.md" ]; then
    check_pass "PLANET_INTEGRATION.md exists"
else
    check_fail "PLANET_INTEGRATION.md not found"
fi

echo ""
echo "2. Checking Python syntax..."
echo ""

# Check Python syntax
if python -m py_compile dreamerv3/nets.py 2>/dev/null; then
    check_pass "dreamerv3/nets.py has valid syntax"
else
    check_fail "dreamerv3/nets.py has syntax errors"
fi

if python -m py_compile dreamerv3/agent.py 2>/dev/null; then
    check_pass "dreamerv3/agent.py has valid syntax"
else
    check_fail "dreamerv3/agent.py has syntax errors"
fi

echo ""
echo "3. Checking PlaNet implementation..."
echo ""

# Check if PlaNet class exists in nets.py
if grep -q "class PlaNet" dreamerv3/nets.py; then
    check_pass "PlaNet class found in nets.py"
else
    check_fail "PlaNet class not found in nets.py"
fi

# Check if required methods exist
REQUIRED_METHODS=("def initial" "def observe" "def imagine" "def obs_step" "def img_step" "def dyn_loss" "def rep_loss")

for method in "${REQUIRED_METHODS[@]}"; do
    if grep -q "$method" dreamerv3/nets.py | head -400 | tail -200; then
        # Check within PlaNet class section
        :  # Method exists
    fi
done

if grep -A 200 "class PlaNet" dreamerv3/nets.py | grep -q "def initial"; then
    check_pass "PlaNet.initial() method exists"
else
    check_warn "PlaNet.initial() method not found"
fi

if grep -A 200 "class PlaNet" dreamerv3/nets.py | grep -q "def observe"; then
    check_pass "PlaNet.observe() method exists"
else
    check_warn "PlaNet.observe() method not found"
fi

if grep -A 200 "class PlaNet" dreamerv3/nets.py | grep -q "def imagine"; then
    check_pass "PlaNet.imagine() method exists"
else
    check_warn "PlaNet.imagine() method not found"
fi

echo ""
echo "4. Checking configuration..."
echo ""

# Check if world_model_type is in config
if grep -q "world_model_type" dreamerv3/dreamerv3.yaml; then
    check_pass "world_model_type configuration found"
else
    check_fail "world_model_type configuration not found"
fi

# Check if planet configs exist
if grep -q "^planet:" dreamerv3/dreamerv3.yaml; then
    check_pass "planet configuration block found"
else
    check_fail "planet configuration block not found"
fi

if grep -q "^planet_small:" dreamerv3/dreamerv3.yaml; then
    check_pass "planet_small configuration found"
else
    check_warn "planet_small configuration not found"
fi

if grep -q "^planet_medium:" dreamerv3/dreamerv3.yaml; then
    check_pass "planet_medium configuration found"
else
    check_warn "planet_medium configuration not found"
fi

echo ""
echo "5. Checking WorldModel integration..."
echo ""

# Check if WorldModel uses world_model_type
if grep -A 20 "class WorldModel" dreamerv3/agent.py | grep -q "world_model_type"; then
    check_pass "WorldModel supports model selection"
else
    check_fail "WorldModel doesn't support model selection"
fi

if grep -A 20 "class WorldModel" dreamerv3/agent.py | grep -q "PlaNet"; then
    check_pass "WorldModel can instantiate PlaNet"
else
    check_fail "WorldModel cannot instantiate PlaNet"
fi

echo ""
echo "6. Checking test scripts..."
echo ""

if [ -f "test_planet_quick.sh" ]; then
    check_pass "test_planet_quick.sh exists"
    if [ -x "test_planet_quick.sh" ]; then
        check_pass "test_planet_quick.sh is executable"
    else
        check_warn "test_planet_quick.sh is not executable (run: chmod +x test_planet_quick.sh)"
    fi
else
    check_warn "test_planet_quick.sh not found"
fi

if [ -f "compare_planet_rssm.sh" ]; then
    check_pass "compare_planet_rssm.sh exists"
else
    check_warn "compare_planet_rssm.sh not found"
fi

if [ -f "analyze_comparison.py" ]; then
    check_pass "analyze_comparison.py exists"
else
    check_warn "analyze_comparison.py not found"
fi

echo ""
echo "7. Checking documentation..."
echo ""

if [ -f "PLANET_INTEGRATION.md" ]; then
    check_pass "PLANET_INTEGRATION.md documentation found"
    # Check if it has key sections
    if grep -q "## 使用方法" PLANET_INTEGRATION.md; then
        check_pass "Usage section found in documentation"
    fi
else
    check_warn "PLANET_INTEGRATION.md not found"
fi

if [ -f "TESTING_GUIDE.md" ]; then
    check_pass "TESTING_GUIDE.md found"
else
    check_warn "TESTING_GUIDE.md not found"
fi

if [ -f "TEST_README.md" ]; then
    check_pass "TEST_README.md found"
else
    check_warn "TEST_README.md not found"
fi

echo ""
echo "=============================================="
echo "Verification Summary"
echo "=============================================="
echo ""
echo -e "${GREEN}Passed:${NC}   $SUCCESS"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILURES"
echo ""

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ PlaNet integration appears to be correctly installed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Read PLANET_INTEGRATION.md for usage instructions"
    echo "  2. Run quick test: ./test_planet_quick.sh 2000 0 carla_four_lane 1000"
    echo "  3. Start training: bash train_dm3.sh 2000 0 --configs planet_medium --task carla_four_lane"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the errors above.${NC}"
    echo ""
    exit 1
fi
