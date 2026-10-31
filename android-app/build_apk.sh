#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$LOCALAPPDATA/Android/Sdk}}"
if [[ ! -d "$SDK/build-tools" ]]; then
  echo "Android SDK not found at: $SDK" >&2
  exit 1
fi

BUILD_TOOLS_VERSION="$(ls "$SDK/build-tools" | sort -V | tail -1)"
BT="$SDK/build-tools/$BUILD_TOOLS_VERSION"
PLATFORM_VERSION="$(ls "$SDK/platforms" | sort -V | tail -1)"
ANDROID_JAR="$SDK/platforms/$PLATFORM_VERSION/android.jar"
JBR="${JBR:-/c/Program Files/Android/Android Studio/jbr}"
JAVAC="$JBR/bin/javac.exe"
JAVA="$JBR/bin/java.exe"
KEYTOOL="$JBR/bin/keytool.exe"

for tool in "$BT/aapt2.exe" "$BT/d8.bat" "$BT/zipalign.exe" "$BT/apksigner.bat" "$ANDROID_JAR" "$JAVAC"; do
  if [[ ! -e "$tool" ]]; then
    echo "Missing build tool: $tool" >&2
    exit 1
  fi
done

export JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
export PATH="$JBR/bin:$PATH"

rm -rf build
mkdir -p build/classes build/dex build/gen

echo "==> Compiling resources"
"$BT/aapt2.exe" compile --dir res -o build/res.zip

echo "==> Linking manifest and assets"
"$BT/aapt2.exe" link \
  -o build/app-unsigned.apk \
  -I "$ANDROID_JAR" \
  --manifest AndroidManifest.xml \
  -A assets \
  --min-sdk-version 24 \
  --target-sdk-version 35 \
  --version-code 2 \
  --version-name "1.1-live-ar" \
  build/res.zip

echo "==> Compiling Java"
find src -name '*.java' > build/sources.txt
"$JAVAC" -source 8 -target 8 -bootclasspath "$ANDROID_JAR" \
  -d build/classes @build/sources.txt

echo "==> Dexing"
find build/classes -name '*.class' > build/classes.txt
cmd //c "$BT\\d8.bat" --release --min-api 24 --lib "$ANDROID_JAR" \
  --output build/dex @"build\\classes.txt"

echo "==> Packaging classes.dex"
python - <<'PY'
import zipfile
with zipfile.ZipFile('build/app-unsigned.apk', 'a', zipfile.ZIP_DEFLATED) as z:
    z.write('build/dex/classes.dex', 'classes.dex')
PY

echo "==> Aligning"
"$BT/zipalign.exe" -f 4 build/app-unsigned.apk build/app-aligned.apk

KEYSTORE="debug.keystore"
if [[ ! -f "$KEYSTORE" ]]; then
  echo "==> Creating debug keystore"
  "$KEYTOOL" -genkeypair -noprompt \
    -keystore "$KEYSTORE" \
    -storepass android -alias androiddebugkey -keypass android \
    -dname "CN=Android Debug,O=Android,C=US" \
    -keyalg RSA -keysize 2048 -validity 10000
fi

echo "==> Signing"
cmd //c "$BT\\apksigner.bat" sign \
  --ks "$KEYSTORE" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out OMR-Checker-pilot.apk \
  build/app-aligned.apk

echo "==> Verifying"
cmd //c "$BT\\apksigner.bat" verify --verbose OMR-Checker-pilot.apk

ls -lh OMR-Checker-pilot.apk
echo "APK: $ROOT/OMR-Checker-pilot.apk"
