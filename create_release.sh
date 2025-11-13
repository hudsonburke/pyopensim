#!/bin/bash
# Script to create a GitHub release which will trigger wheel building

set -e

# Determine version from OpenSim submodule
cd src/opensim-core
OPENSIM_VERSION=$(git describe --tags | grep -oP '^\d+\.\d+\.\d+')
cd ../..

# Set build number (increment this for subsequent releases of same OpenSim version)
BUILD_NUMBER=${BUILD_NUMBER:-0}
VERSION="${OPENSIM_VERSION}.${BUILD_NUMBER}"

echo "=================================================="
echo "Creating PyOpenSim Release"
echo "=================================================="
echo "OpenSim Version: $OPENSIM_VERSION"
echo "Build Number: $BUILD_NUMBER"
echo "Full Version: $VERSION"
echo "=================================================="

# Check if tag already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "⚠️  Tag v$VERSION already exists!"
    echo "To create a new release, increment BUILD_NUMBER:"
    echo "  BUILD_NUMBER=1 $0"
    exit 1
fi

# Commit any changes
echo "Checking for uncommitted changes..."
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "📝 You have uncommitted changes. Committing them..."
    git add -A
    git commit -m "chore: Prepare release v$VERSION"
else
    echo "✓ No uncommitted changes"
fi

# Push changes
echo "Pushing to GitHub..."
git push origin main

# Create and push tag
echo "Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION

PyOpenSim $VERSION

- OpenSim Core: $OPENSIM_VERSION
- Build: $BUILD_NUMBER
"

git push origin "v$VERSION"

echo ""
echo "=================================================="
echo "✅ Tag v$VERSION created and pushed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/hudsonburke/pyopensim/releases/new?tag=v$VERSION"
echo "2. Click 'Generate release notes' to auto-populate changelog"
echo "3. Add any additional notes about this release"
echo "4. Click 'Publish release'"
echo ""
echo "The GitHub Actions workflow will automatically:"
echo "  - Build wheels for Linux, macOS, and Windows"
echo "  - Upload wheels to PyPI"
echo ""
echo "Monitor progress at:"
echo "https://github.com/hudsonburke/pyopensim/actions"
echo ""
