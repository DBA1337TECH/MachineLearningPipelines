GHIDRA_HOME=/home/bdg/Downloads/ghidra_11.2.1_PUBLIC_20241105/ghidra_11.2.1_PUBLIC/

$GHIDRA_HOME/support/analyzeHeadless \
  /home/bdg/ \
  router \
  -process "CISCO/cpio-root/asa/bin/lina" \
  -scriptPath /home/bdg/ghidra_scripts \
  -postScript ExtractFeatures

